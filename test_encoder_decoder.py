import argparse
import os
import torch
from pretext_task.pretext_models_FPN import FPNBasedEncoder, FPNBasedDecoderInterpolate, Discriminator
from downstream_task.downstream_models import DownstreamLinear
from torch.utils.data import DataLoader
from torchvision import transforms
from pretext_task.dataset import TestDataset
from downstream_task.dataset import DatasetDownstreamTestADE
from check_encoder_decoder import OperatorPretext
from configs import cfg
from utils import find_recursive
from pretext_task.pretext_models_deeplabV3pluse import DeepLabV3PlusEncoder, DecoderDeeplabBased
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
                                    transforms.Resize((cfg.DATASET_PRE.crop_size,cfg.DATASET_PRE.crop_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                    ])
        testdata = TestDataset(cfg.DATASET_TEST, trans)
        testloader = DataLoader(testdata, batch_size=1)

        decoder = DecoderDeeplabBased({"depth": 1, "norm": 3, "edge": 1})
        decoder.load_state_dict(torch.load(cfg.TEST.decoder))

        operator = OperatorPretext(cfg.TRAIN_PRE, encoder, decoder, netD=Discriminator(), dataloader=testloader)
        operator.test()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--imgs",
            required=True,
            type=str,
            help="an image path, or a directory name"
        )
        args = parser.parse_args()

        # generate testing image list
        if os.path.isdir(args.imgs):
            imgs = find_recursive(args.imgs)
        else:
            imgs = [args.imgs]
        assert len(imgs), "imgs should be a path to image (.jpg) or directory."
        cfg.list_test = [{'fpath_img': x} for x in imgs]

        # utilize test dataset for downstream task
        testdata = DatasetDownstreamTestADE(cfg.list_test, cfg.TEST)
        testloader = DataLoader(testdata, batch_size=1)
        # downstream task testing
        decoder = DownstreamLinear(cfg.DATASET_DOWN.num_class)
        decoder.load_state_dict(torch.load(cfg.TEST.decoder_down))
        operator = OperatorDownstream(cfg.TRAIN_DOWN, encoder, decoder, testloader)
        operator.test(cfg.TEST.results)


if __name__ == "__main__":
    main()
