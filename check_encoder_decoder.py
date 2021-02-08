import os
import time
import cv2
import torch
import numpy as np
import wandb
from utils import InitPre, save_models, save_checkpoints, add_gaussian_noise, InitDown, visualize_result, AverageMeter, \
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
            for idx, syn_dict in enumerate(self.dataloader[0]):
                print(f'epoch:{epoch}, iter:{idx}')
                # get inputs
                syn_imgs = syn_dict['color'].to(self.device)
                syn_nor = syn_dict['normal'].to(self.device)
                syn_edge = syn_dict['edge'].to(self.device)
                syn_depth = syn_dict['depth'].to(self.device)
                check_isfinite("syn_depth.sum()", syn_depth.sum())
                syn_edge_count = syn_dict['edge_c'].to(self.device)

                encoder_out_syn = self.encoder(syn_imgs)

                self.opt_encoder.zero_grad()
                self.opt_decoder.zero_grad()

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

                loss = loss_dep + loss_edge + loss_norm
                loss.backward()
                self.opt_encoder.step()
                self.opt_decoder.step()

                if idx % 10 == 0:
                    wandb.log({"loss_dep": loss_dep, "loss_edge": loss_edge, "loss_norm": loss_norm,
                               "loss": loss})

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
            de_out = self.decoder(en_out[0],en_out[1])
            norm = de_out['normal'].squeeze().cpu().detach().numpy() * 255.0
            norm = np.transpose(norm, [1, 2, 0])
            depth = de_out['depth'].squeeze().cpu().detach().numpy()
            depth = np.exp(depth) * 1000.0
            depth = cv2.normalize(depth, depth, cv2.NORM_MINMAX) * 255
            edge = de_out['edge'].squeeze().cpu().detach().numpy()
            edge[edge > 0.5] = 255
            edge[edge <= 0.5] = 0
            norm = cv2.cvtColor(norm, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join('pretext_task/outputs/norm', 'norm_' + img_name + '.png'), norm.astype(np.int))
            cv2.imwrite(os.path.join('pretext_task/outputs/edge', 'edge_' + img_name + '.png'), edge)
            cv2.imwrite(os.path.join('pretext_task/outputs/depth', 'depth_' + img_name + '.png'), depth.astype(np.int))


def check_isfinite(name, params):
    print(params)
    print(f"params name: {name}, is_finite: {torch.isfinite(params)}")
