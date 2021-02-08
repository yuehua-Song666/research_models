import os
import time
import cv2
import torch
import numpy as np
import wandb
from utils import InitPre, save_models, save_checkpoints, add_gaussian_noise, InitDown, visualize_result, AverageMeter,\
    accuracy, intersectionAndUnion, summary


class OperatorPretext:
    def __init__(self, cfg, encoder, decoder, netD, dataloader):
        self.cfg = cfg
        self.dataloader = dataloader
        self.std = 1.5
        torch.autograd.set_detect_anomaly(True)
        init = InitPre(cfg=cfg)
        # init models
        self.encoder, self.decoder, self.discriminator = init.init_models(encoder, decoder, netD)
        if cfg.train:
            self.start = 0
            self.epoch = self.cfg.epoch
            # init optims
            self.opt_encoder, self.opt_decoder, self.opt_dis, self.scheduler = init.init_optimizers(
                encoder=self.encoder,
                decoder=self.decoder,
                discriminator=self.discriminator)
            # init criterion
            self.criterion_gan, self.criterion_norm, self.criterion_depth, self.criterion_edge = init.init_criterions()

            wandb.init(project='synthetic-real-feature-adaption')
            wandb.watch(self.encoder)
            wandb.watch(self.decoder)
            wandb.watch(self.discriminator)

        if cfg.train and cfg.resume:
            checkpoint = torch.load(self.cfg.checkpoint_path)
            self.start = checkpoint['epoch'] + 1
            self.std = checkpoint['std']

        self.device = init.check_cuda()

    def train(self):
        for epoch in range(self.start, self.epoch):
            for idx, (real_sample, syn_dict) in enumerate(zip(self.dataloader[0], self.dataloader[1])):
                print(f'epoch:{epoch}, iter:{idx}')
                # get inputs
                batch_size = real_sample.size(0)
                real_imgs = real_sample.to(self.device)
                syn_imgs = syn_dict['color'].to(self.device)
                syn_nor = syn_dict['normal'].to(self.device)
                syn_edge = syn_dict['edge'].to(self.device)
                syn_depth = syn_dict['depth'].to(self.device)
                check_isfinite("syn_depth.sum()", syn_depth.sum())
                syn_edge_count = syn_dict['edge_c'].to(self.device)

                # create label
                if idx % 3 == 0:
                    label = torch.full([batch_size, ], 0.0, device=self.device)
                else:
                    label = torch.full([batch_size, ], 1.0, device=self.device)

                encoder_out_syn = self.encoder(syn_imgs)

                # forward: update discriminator
                for p in self.discriminator.parameters():
                    p.requires_grad = True
                self.opt_dis.zero_grad()

                # syn samples
                out_syn = add_gaussian_noise(encoder_out_syn[-1].detach(), self.std, self.device)
                out_dis_syn = self.discriminator(out_syn)
                loss_discriminator_real = self.criterion_gan(out_dis_syn, label)
                loss_discriminator_real.backward()

                # real samples
                encoder_out_real = self.encoder(real_imgs)
                if idx % 3 == 0:
                    label.fill_(1.0)
                else:
                    label.fill_(0.0)

                out_real = add_gaussian_noise(encoder_out_real[-1].detach(), self.std, self.device)
                out_dis_real = self.discriminator(out_real)
                loss_discriminator_fake = self.criterion_gan(out_dis_real, label)
                loss_discriminator_fake.backward()
                self.opt_dis.step()

                # forward: update encoder and decoder
                # set discriminator parameters false and clean grads
                for p in self.discriminator.parameters():
                    p.requires_grad = False
                self.opt_encoder.zero_grad()
                self.opt_decoder.zero_grad()

                # step 1: encoder loss
                loss_en = self.criterion_gan(self.discriminator(encoder_out_syn[-1]), label)

                # step2: decoder loss
                # the following part from https://github.com/jason718/game-feature-learning
                # decoder_out = self.decoder(*encoder_out_syn[:3], encoder_out_syn[4])
                decoder_out = self.decoder(*encoder_out_syn)

                # depth
                depth_diff = decoder_out['depth'] - syn_depth
                _n = decoder_out['depth'].size(0) * decoder_out['depth'].size(2) * decoder_out['depth'].size(3)
                # check_isfinite("diff.sum", depth_diff.sum())
                # check_isfinite("diff.sum.div", depth_diff.sum().div_(_n*_n))
                # check_isfinite("diff.sum.div.mul", depth_diff.sum().div_(_n*_n).mul_(0.5))
                loss_depth2 = depth_diff.sum().div_(_n * _n).mul_(0.5)
                loss_depth1 = self.criterion_depth(decoder_out['depth'], syn_depth)
                # loss_dep = self.cfg['DEP_WEIGHT'] * loss_depth1
                loss_dep = self.cfg.depth_weight * (loss_depth1 + loss_depth2) * 0.5

                # surface normal
                pred_norm = decoder_out['normal']
                loss_norm = self.cfg.norm_weight * self.criterion_norm(pred_norm, syn_nor)

                # edge
                # weight_e = (decoder_out['edge'].size(2) * decoder_out['edge'].size(
                #     3) - syn_edge_count) / syn_edge_count
                # self.criterionEdge = torch.nn.BCEWithLogitsLoss(weight=weight_e.float().view(-1, 1, 1, 1)).to(
                #     self.device)
                loss_edge = self.cfg.edge_weight * self.criterion_edge(decoder_out['edge'], syn_edge,
                                                                       syn_edge_count)

                loss = loss_dep + loss_edge + loss_norm + loss_en * self.cfg.da_weight
                loss.backward()
                self.opt_encoder.step()
                self.opt_decoder.step()

                if idx % 10 == 0:
                    wandb.log({"loss_dep": loss_dep, "loss_edge": loss_edge, "loss_norm": loss_norm, "loss_en": loss_en,
                               "loss": loss, "dis_real": loss_discriminator_real, "dis_fake": loss_discriminator_fake})

            self.std -= self.std / self.epoch
            self.scheduler.step()
            # save models and checkpoints on every epoch
            save_models(self.encoder, self.decoder, self.discriminator, epoch, self.cfg.save_path)
            save_checkpoints({'encoder': self.encoder, 'decoder': self.decoder, 'discriminator': self.discriminator},
                             {'opt_encoder': self.opt_encoder, 'opt_decoder': self.opt_decoder,
                              'opt_dis': self.opt_dis},
                             epoch, self.std, self.cfg.checkpoint_path)

    # Trying the new idea
    def newtrain(self):
        for epoch in range(self.start, self.epoch):
            for idx, (real_sample, syn_dict) in enumerate(zip(self.dataloader[0], self.dataloader[1])):
                print(f'epoch:{epoch}, iter:{idx}')
                # get inputs
                batch_size = real_sample.size(0)
                real_imgs = real_sample.to(self.device)
                syn_imgs = syn_dict['color'].to(self.device)
                syn_nor = syn_dict['normal'].to(self.device)
                syn_edge = syn_dict['edge'].to(self.device)
                syn_instance = syn_dict['instance'].to(self.device)
                # check_isfinite("syn_depth.sum()", syn_depth.sum())
                syn_edge_count = syn_dict['edge_c'].to(self.device)

                # create label
                if idx % 3 == 0:
                    label = torch.full([batch_size, ], 0.0, device=self.device)
                else:
                    label = torch.full([batch_size, ], 1.0, device=self.device)

                encoder_out_syn = self.encoder(syn_imgs)

                # forward: update discriminator
                for p in self.discriminator.parameters():
                    p.requires_grad = True
                self.opt_dis.zero_grad()

                # syn samples
                out_syn = add_gaussian_noise(encoder_out_syn[-1].detach(), self.std, self.device)
                # Add decoder here
                decoder_out_syn = self.decoder(*encoder_out_syn)
                norm_syn = decoder_out_syn['normal']
                edge_syn = decoder_out_syn['edge']
                out_dis_syn_norm = self.discriminator(norm_syn.detach())
                out_dis_syn_edge = self.discriminator(edge_syn, is_edge = True)
                loss_discriminator_fake_norm = self.criterion_gan(out_dis_syn_norm, label)
                loss_discriminator_fake_edge = self.criterion_gan(out_dis_syn_edge, label)
                loss_discriminator_fake_norm.backward(retain_graph=True)
                loss_discriminator_fake_edge.backward(retain_graph=True)

                # real samples
                encoder_out_real = self.encoder(real_imgs)
                if idx % 3 == 0:
                    label.fill_(1.0)
                else:
                    label.fill_(0.0)

                out_real = add_gaussian_noise(encoder_out_real[-1].detach(), self.std, self.device)
                decoder_out_real = self.decoder(*encoder_out_real)
                norm_real = decoder_out_real['normal']
                edge_real = decoder_out_real['edge']
                out_dis_real_norm = self.discriminator(norm_real.detach())
                out_dis_real_edge = self.discriminator(edge_real, is_edge = True)
                loss_discriminator_real_norm = self.criterion_gan(out_dis_real_norm, label)
                loss_discriminator_real_edge = self.criterion_gan(out_dis_real_edge, label)
                loss_discriminator_real_norm.backward(retain_graph=True)
                loss_discriminator_real_edge.backward(retain_graph=True)
                self.opt_dis.step()

                # forward: update encoder and decoder
                # set discriminator parameters false and clean grads
                for p in self.discriminator.parameters():
                    p.requires_grad = False
                self.opt_encoder.zero_grad()
                self.opt_decoder.zero_grad()

                # loss_en = self.criterion_gan(self.discriminator(encoder_out_syn[-1]), label)

                pred_norm = decoder_out_syn['normal']
                loss_norm = self.cfg.norm_weight * self.criterion_norm(pred_norm, syn_nor)

                loss_edge = self.cfg.edge_weight * self.criterion_edge(decoder_out_syn['edge'], syn_edge, syn_edge_count)

                # loss = loss_edge + loss_norm + loss_en * self.cfg.da_weight
                loss = loss_norm + loss_edge
                loss.backward()
                self.opt_encoder.step()
                self.opt_decoder.step()

                if idx % 10 == 0:
                    wandb.log({# "loss_edge": loss_edge,
                               "loss_norm": loss_norm,
                               # "loss_en": loss_en,
                               "loss": loss,
                               "dis_real_norm": loss_discriminator_real_norm,
                               # "dis_real_edge": loss_discriminator_real_edge,
                               "dis_fake_norm": loss_discriminator_fake_norm,
                               # "dis_fake_edge": loss_discriminator_fake_edge
                               })

            self.std -= self.std / self.epoch
            self.scheduler.step()
            # save models and checkpoints on every epoch
            save_models(self.encoder, self.decoder, self.discriminator, epoch, self.cfg.save_path)
            save_checkpoints({'encoder': self.encoder, 'decoder': self.decoder, 'discriminator': self.discriminator},
                             {'opt_encoder': self.opt_encoder, 'opt_decoder': self.opt_decoder,
                              'opt_dis': self.opt_dis},
                             epoch, self.std, self.cfg.checkpoint_path)

    def test(self):
        for idx, data in enumerate(self.dataloader):
            img_name = str(data[0][0])
            img_name = img_name.translate(str.maketrans('', '', '.jpg'))
            img = data[1].to(self.device)
            en_out = self.encoder(img)
            de_out = self.decoder(*en_out[:3], en_out[4])
            norm = de_out['normal'].squeeze().cpu().detach().numpy() * 255.0
            norm = np.transpose(norm, [1, 2, 0])
            # depth = de_out['depth'].squeeze().cpu().detach().numpy()
            # depth = np.exp(depth) * 1000.0
            # depth = cv2.normalize(depth, depth, cv2.NORM_MINMAX) * 255
            edge = de_out['edge'].squeeze().cpu().detach().numpy()
            edge[edge > 0.5] = 255
            edge[edge <= 0.5] = 0
            norm = cv2.cvtColor(norm, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join('pretext_task/outputs_result_norm/norm', 'norm_' + img_name + '.png'), norm.astype(np.int))
            cv2.imwrite(os.path.join('pretext_task/outputs_result_norm/edge', 'edge_' + img_name + '.png'), edge)
            # cv2.imwrite(os.path.join('pretext_task/outputs/depth', 'depth_' + img_name + '.png'), depth.astype(np.int))

    def newtest(self):
        for idx, data in enumerate(self.dataloader):
            img_name = str(data[0][0])
            img_name = img_name.translate(str.maketrans('', '', '.jpg'))
            img = data[1].to(self.device)
            en_out = self.encoder(img)
            de_out = self.decoder(en_out[0], en_out[1])
            norm = de_out['normal'].squeeze().cpu().detach().numpy() * 255.0
            norm = np.transpose(norm, [1, 2, 0])
            # depth = de_out['depth'].squeeze().cpu().detach().numpy()
            # depth = np.exp(depth) * 1000.0
            # depth = cv2.normalize(depth, depth, cv2.NORM_MINMAX) * 255
            edge = de_out['edge'].squeeze().cpu().detach().numpy()
            edge[edge > 0.5] = 255
            edge[edge <= 0.5] = 0
            norm = cv2.cvtColor(norm, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join('pretext_task/outputs_result_norm/norm', 'norm_' + img_name + '.png'), norm.astype(np.int))
            cv2.imwrite(os.path.join('pretext_task/outputs_result_norm/edge', 'edge_' + img_name + '.png'), edge)
            # cv2.imwrite(os.path.join('pretext_task/outputs/depth', 'depth_' + img_name + '.png'), depth.astype(np.int))


class OperatorDownstream:
    def __init__(self, cfg, encoder, decoder, dataloader):
        init = InitDown(cfg)
        self.cfg = cfg
        self.dataloader = dataloader
        self.encoder, self.decoder = init.init_models(encoder, decoder)
        self.device = init.check_cuda()
        if self.cfg.train:
            self.optimizer = init.init_optimizers(decoder)
            self.criterion = init.init_criterions()
            wandb.init(project='synthetic-real-feature-adaption')
            wandb.watch(decoder)

    def train(self):
        for epoch in range(self.cfg.epoch):
            for idx, (img, sgm) in enumerate(self.dataloader):
                print(f"epoch:{epoch}, iter:{idx}")
                img = img.to(self.device)
                sgm = sgm.to(self.device)
                self.decoder.zero_grad()
                # pass to encoder, decoder
                out = self.decoder(*self.encoder(img)[:-1])
                loss = self.criterion(out, sgm)
                loss.backward()
                self.optimizer.step()
                wandb.log({"loss": loss})
        torch.save(self.decoder.state_dict(), os.path.join(self.cfg.save_path, "model.pkl"))

    def newtrain(self):
        for epoch in range(self.cfg.epoch):
            for idx, (img, sgm) in enumerate(self.dataloader):
                print(f"epoch:{epoch}, iter:{idx}")
                img = img.to(self.device)
                # sgm = sgm.float()
                sgm = sgm.to(self.device)

                self.decoder.zero_grad()
                # pass to encoder, decoder
                out = self.decoder(*self.encoder(img))
                norm_pred = out['normal']
                loss = self.criterion(norm_pred, sgm)
                loss.backward(retain_graph=True)
                self.optimizer.step()
                wandb.log({"loss": loss})
        torch.save(self.decoder.state_dict(), os.path.join(self.cfg.save_path, "model.pkl"))

    def test(self, save_path):
        for data in self.dataloader:
            pred = self.decoder(*self.encoder(data["img"].to(self.device))[:-1])
            _, pred = torch.max(pred, dim=1)
            pred = np.asarray(pred.squeeze(0).cpu())
            visualize_result(data, pred, save_path)

    def eval(self):
        acc_meter = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        time_meter = AverageMeter()
        for data in self.dataloader:
            tic = time.perf_counter()
            seg_label = np.asarray(data["segm"])
            pred = self.decoder(*self.encoder(data["img"].to(self.device))[:-1])
            _, pred = torch.max(pred, dim=1)
            pred = np.asarray(pred.squeeze(0).cpu())
            time_meter.update(time.perf_counter() - tic)

            # calculate accuracy
            acc, pix = accuracy(pred, seg_label)
            intersection, union = intersectionAndUnion(pred, seg_label, self.cfg.num_class)
            acc_meter.update(acc, pix)
            intersection_meter.update(intersection)
            union_meter.update(union)

        # summary
        summary(intersection_meter, union_meter, acc_meter, time_meter)


def check_isfinite(name, params):
    print(params)
    print(f"params name: {name}, is_finite: {torch.isfinite(params)}")