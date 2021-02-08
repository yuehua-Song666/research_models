import argparse
import os
import torch
from pretext_task.pretext_models_FPN import FPNBasedEncoder, FPNBasedDecoderInterpolate, Discriminator, Discriminator_after_decoder
from downstream_task import DatasetDownstreamTrainADE, DownstreamLinear, DatasetDownstreamTrainVoc
from torch.utils.data import DataLoader
from torchvision import transforms
from pretext_task.dataset import TestDataset
from downstream_task.dataset import DatasetDownstreamTestADE
from operate import OperatorPretext, OperatorDownstream
from configs import cfg
from pretext_task.pretext_models_deeplabV3pluse import DeepLabV3PlusEncoder, DecoderDeeplabBased

from utils import find_recursive
# from FPNAttentionModels.SKNet import FPNbBasedEncoder
# from FPNAttentionModels.CBAM import FPNbBasedEncoder


def main():
    # load encoder
    # encoder = FPNBasedEncoder('resnet50')
    encoder = DeepLabV3PlusEncoder(middle_times=16, exit_strid=1, exit_dilation=2)
    # load weights
    encoder.load_state_dict(torch.load(cfg.TEST.encoder))
    if cfg.pretext:
        # pretext task testing
        # load data
        trans = transforms.Compose([transforms.ToPILImage(mode='RGB'),
                                    # transforms.Resize(cfg.DATASET_PRE.crop_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                    ])
        testdata = TestDataset(cfg.DATASET_TEST, trans)
        testloader = DataLoader(testdata, batch_size=1)

        # decoder = FPNBasedDecoderInterpolate()
        decoder = DecoderDeeplabBased({"norm": 3, "edge": 1})
        decoder.load_state_dict(torch.load(cfg.TEST.decoder))

        operator = OperatorPretext(cfg.TRAIN_PRE, encoder, decoder, netD=Discriminator_after_decoder(), dataloader=testloader)
        operator.newtest()



if __name__ == "__main__":
    main()