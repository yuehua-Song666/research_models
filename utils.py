import csv
import fnmatch
import os
from functools import wraps
import cv2
import torch
import numpy as np
from PIL import Image
from scipy.io import loadmat
from torch import nn
from torch.optim import lr_scheduler
from skimage import transform

colors = loadmat('configs/color150.mat')['colors']
names = {}
with open('configs/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


# adjust lr, call once per epoch
class GetScheduler:
    def __init__(self, optimizers):
        self.optimizers = optimizers
        scheduler_en = lr_scheduler.MultiStepLR(*optimizers['encoder'])
        scheduler_de = lr_scheduler.MultiStepLR(*optimizers['decoder'])
        scheduler_dis = lr_scheduler.MultiStepLR(*optimizers['dis'])
        self.schedulers = [scheduler_en, scheduler_de, scheduler_dis]

    def step(self):
        for scheduler, (opt_name, optimizer) in zip(self.schedulers, self.optimizers.items()):
            scheduler.step()
            lr = optimizer[0].param_groups[0]['lr']
            print('learning rate, %s = %.7f' % (opt_name, lr))


class Rescale(object):
    """ Rescale the image in a sample to a given size. """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        # color_img, depth_img, edge_img, normal_img = sample['color'], sample['depth'], sample['edge'], sample['normal']
        color_img, instance_img, edge_img, normal_img = sample['color'], sample['instance'], sample['edge'], sample['normal']

        # color_im = cv2.resize(color_img, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)
        # depth_im = cv2.resize(depth_img, (self.output_size, self.output_size), interpolation=cv2.INTER_CUBIC)
        # normal_im = cv2.resize(normal_img, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)
        # edge_im = cv2.resize(edge_img, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)
        color_im = transform.resize(color_img, (self.output_size, self.output_size), mode='reflect',
                                    preserve_range=True, anti_aliasing=True)
        # depth_im = transform.resize(depth_img, (self.output_size, self.output_size), mode='reflect',
        #                             preserve_range=True, anti_aliasing=True)
        instance_im = transform.resize(instance_img, (self.output_size, self.output_size), mode='reflect',
                                   preserve_range=True, anti_aliasing=True)
        normal_im = transform.resize(normal_img, (self.output_size, self.output_size), mode='reflect',
                                     preserve_range=True, anti_aliasing=True)
        edge_im = transform.resize(edge_img, (self.output_size, self.output_size), mode='reflect',
                                   preserve_range=True, anti_aliasing=True)
        # print("ndarray_rescale", depth_im.sum(), np.max(depth_im))
        # return {'color': color_im, 'depth': depth_im, 'edge': edge_im, 'normal': normal_im}
        return {'color': color_im, 'instance': instance_im, 'edge': edge_im, 'normal': normal_im}


class RandomCrop(object):
    """ Crop randomly the image in a sample. """

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = (output_size, output_size)

    def __call__(self, sample):
        color_img, depth_img, edge_img, normal_img = sample['color'], sample['depth'], sample['edge'], sample['normal']

        h, w = color_img.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        color_im = color_img[top: top + new_h, left: left + new_w]
        depth_im = depth_img[top: top + new_h, left: left + new_w]
        edge_im = edge_img[top: top + new_h, left: left + new_w]
        normal_im = normal_img[top: top + new_h, left: left + new_w]

        return {'color': color_im, 'depth': depth_im, 'edge': edge_im, 'edge_pix': sample['edge_pix'],
                'normal': normal_im}


class ToTensor(object):
    """ Convert ndarrays in sample to Tensors. """

    def __call__(self, sample):
        # color_img, depth_img, edge_img, normal_img = sample['color'], sample['depth'], sample['edge'], sample['normal']
        color_img, depth_img, edge_img, normal_img = sample['color'], sample['instance'], sample['edge'], sample['normal']
        # print("ndarray_totensor", depth_img.sum(), np.max(depth_img))
        # swap color axis because
        # numpy image: H x W x C  --> torch image: C x H x W (C=1,3)
        color_img = color_img.transpose((2, 0, 1))
        normal_img = normal_img.transpose((2, 0, 1))
        h, w = depth_img.shape
        depth_img = depth_img.reshape((1, h, w))
        edge_img = edge_img.reshape((1, h, w))
        # print("tensor", torch.from_numpy(depth_img.astype('float32')).sum())
        return {'color': torch.from_numpy(color_img.astype('float32')),
                # 'depth': torch.from_numpy(depth_img.astype('float32')),
                'instance': torch.from_numpy(depth_img.astype('float32')),
                'edge': torch.from_numpy(edge_img.astype('float32')),
                'normal': torch.from_numpy(normal_img.astype('float32'))}


class InitPre:
    def __init__(self, cfg):
        self.train = cfg.train
        self.cfg = cfg
        self.device = self.check_cuda()
        if self.cfg.resume:
            self.checkpoint = torch.load(self.cfg.checkpoint_path)

    def init_models(self, encoder, decoder, discriminator):

        # train mode
        if self.train:
            print('Train Mode')
            encoder.train()
            decoder.train()
            discriminator.train()

            decoder.to(self.device)
            encoder.to(self.device)
            discriminator.to(self.device)

            decoder.apply(_init_weight)
            encoder.apply(_init_weight)
            discriminator.apply(_init_weight)
        # eval mode
        else:
            print('Eval Mode')
            encoder.eval()
            decoder.eval()

            decoder.to(self.device)
            encoder.to(self.device)

        if self.train and self.cfg.resume:
            print('resume training')
            # load models checkoint
            encoder.load_state_dict(self.checkpoint['encoder'])
            decoder.load_state_dict(self.checkpoint['decoder'])
            discriminator.load_state_dict(self.checkpoint['discriminator'])
            print('loaded models successfilly')

        return encoder, decoder, discriminator

    def init_optimizers(self, encoder, decoder, discriminator):
        # define optimizers
        opt_encoder = torch.optim.Adam(encoder.parameters(), lr=self.cfg.lr, betas=(0.5, 0.999))
        # opt_decoder = torch.optim.Adam(decoder.parameters(), lr=self.cfg.lr)
        opt_decoder = torch.optim.Adam(decoder.parameters(), lr=self.cfg.lr, betas=(0.5, 0.999))
        opt_dis = torch.optim.Adam(discriminator.parameters(), lr=self.cfg.lr_d,
                                   betas=(0.5, 0.999))
        scheduler = GetScheduler(
            optimizers={'encoder': [opt_encoder, self.cfg.milestones_en, self.cfg.gamma_en],
                        'decoder': [opt_decoder, self.cfg.milestones_de, self.cfg.gamma_de],
                        'dis': [opt_dis, self.cfg.milestones_dis, self.cfg.gamma_dis]})

        if self.cfg.resume:
            opt_encoder.load_state_dict(self.checkpoint['opt_encoder'])
            opt_decoder.load_state_dict(self.checkpoint['opt_decoder'])
            opt_dis.load_state_dict(self.checkpoint['opt_dis'])
            print('loaded optimizers successfuly')

        return opt_encoder, opt_decoder, opt_dis, scheduler

    @staticmethod
    def init_criterions():
        # define losses
        criterion_gan = nn.BCELoss()
        criterion_depth = nn.MSELoss()
        # self.criterion_norm = nn.CosineEmbeddingLoss()
        criterion_norm = nn.MSELoss()
        # criterion_edge = nn.BCELoss()
        criterion_edge = WeightedBCELoss()

        return criterion_gan, criterion_norm, criterion_depth, criterion_edge

    def check_cuda(self):
        if torch.cuda.is_available():
            device = self.cfg.gpu
        else:
            device = 'cpu'
            print('No GPU Found')

        return device


class InitDown:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = self.check_cuda()

    def init_models(self, encoder, decoder):
        if self.cfg.train:
            encoder.eval()
            # for p in encoder.parameters():
            #     p.require_grads = False
            decoder.train()
        else:
            encoder.eval()
            decoder.eval()

        return encoder.to(self.device), decoder.to(self.device)

    def init_optimizers(self, decoder):
        opt = torch.optim.Adam(decoder.parameters(), lr=self.cfg.lr)

        return opt

    @staticmethod
    def init_criterions():
        cri = nn.CrossEntropyLoss(ignore_index=-1)

        return cri

    def check_cuda(self):
        if torch.cuda.is_available():
            device = self.cfg.gpu
        else:
            device = 'cpu'
            print('No GPU Found')

        return device


# define weighted loss function, according to https://arxiv.org/pdf/1504.06375.pdf
class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()

    def forward(self, inputs, targets, num_pos):
        x = inputs
        z = targets
        num_neg = inputs.size(2) * inputs.size(3) - num_pos
        beta = num_neg / (num_neg + num_pos)
        beta = torch.reshape(beta, [inputs.size(0), 1, 1, 1])
        q = beta / (1 - beta)
        loss = (1 - z) * x + (1 + (q - 1) * z) * (
                torch.log(1 + torch.exp(-torch.abs(x))) + torch.max(-x, torch.zeros(x.size(), device='cuda:0')))
        loss = loss * (1 - beta)

        return torch.mean(loss)


def _init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


def save_models(encoder, decoder, discriminator, epoch, path):
    torch.save(encoder.state_dict(), os.path.join(path, 'encoder', f'EncoderE{epoch}.pkl'))
    torch.save(decoder.state_dict(), os.path.join(path, 'decoder', f'DecoderE{epoch}.pkl'))
    torch.save(discriminator.state_dict(), os.path.join(path, 'discriminator', f'discriminatorE{epoch}.pkl'))
    print(f'save models successfully at path:{path}')


def save_checkpoints(models, optims, epoch, std, path):
    torch.save({
        'epoch': epoch,
        'std': std,
        'encoder': models['encoder'].state_dict(),
        'decoder': models['decoder'].state_dict(),
        'discriminator': models['discriminator'].state_dict(),
        'opt_encoder': optims['opt_encoder'].state_dict(),
        'opt_decoder': optims['opt_decoder'].state_dict(),
        'opt_dis': optims['opt_dis'].state_dict(),
    }, path)


def add_gaussian_noise(input, std, device, mean=0):
    noise = torch.zeros(input.size()[2:], device=device).normal_(mean, std)
    return input + noise.repeat([input.size(0), input.size(1), 1, 1])


def find_recursive(root_dir, ext='.jpg'):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            files.append(os.path.join(root, filename))
    return files


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
                        np.tile(colors[label],
                                (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


# todo: mark the class name
def visualize_result(data, pred, save_path):
    img_path = data["img_path"][0]
    ori_img = data["ori"]
    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    print("Predictions in [{}]:".format(img_path))
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    im_vis = np.concatenate((np.squeeze(ori_img, axis=0), pred_color), axis=1)

    img_name = img_path.split('/')[-1]
    Image.fromarray(im_vis).save(
        os.path.join(save_path, img_name.replace('.jpg', '.png')))


def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)


def summary(intersection_meter, union_meter, acc_meter, time_meter):
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(names[i + 1], _iou))

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean(), acc_meter.average() * 100, time_meter.average()))
