import torch
from configs import cfg
from downstream_task import DatasetDownstreamTrainADE, DownstreamLinear, DatasetDownstreamTrainVoc
# from FPNAttentionModels.CBAM import FPNbBasedEncoder
# from FPNAttentionModels.SKNet import FPNbBasedEncoder
from pretext_task.pretext_models_FPN import FPNBasedEncoder, FPNBasedDecoder, Discriminator, FPNBasedDecoderDilated, \
    FPNBasedDecoderInterpolate
from pretext_task.pretext_models_deeplabV3pluse import DeepLabV3PlusEncoder, DecoderDeeplabBased
from operate import OperatorPretext, OperatorDownstream
from torch.utils.data import DataLoader
from pretext_task.dataset import SynDataset, RealDataset


def main():
    if cfg.pretext:
        # pretext task training
        # creat real image loader
        real_dataset = RealDataset(cfg.DATASET_PRE)

        real_dataloader = DataLoader(real_dataset, batch_size=cfg.TRAIN_PRE.batch_size, shuffle=True, num_workers=16)

        # create syntehtic image loader
        syn_dataset = SynDataset(cfg.DATASET_PRE)
        syn_dataloader = DataLoader(syn_dataset, batch_size=cfg.TRAIN_PRE.batch_size, shuffle=True, drop_last=True,
                                    num_workers=16)

        # load model
        # encoder = FPNBasedEncoder('resnet50', se=False)
        encoder = DeepLabV3PlusEncoder(middle_times=16, exit_strid=1, exit_dilation=2)
        # decoder = FPNBasedDecoderInterpolate()
        decoder = DecoderDeeplabBased({"depth": 1, "norm": 3, "edge": 1})
        netD = Discriminator()

        operate = OperatorPretext(cfg.TRAIN_PRE, encoder=encoder, decoder=decoder, netD=netD,
                                  dataloader=[real_dataloader, syn_dataloader])
        operate.train()
    else:
        # downstream task training
        dataset = DatasetDownstreamTrainADE(cfg.DATASET_DOWN.list_train, cfg.DATASET_DOWN,
                                            max_sample=10000)
        loader = DataLoader(dataset=dataset, batch_size=cfg.TRAIN_DOWN.batch_size, shuffle=True, num_workers=8)

        encoder = FPNBasedEncoder("resnet50")
        encoder.load_state_dict(torch.load(cfg.TEST.encoder))
        decoder = DownstreamLinear(cfg.DATASET_DOWN.num_class)

        operator = OperatorDownstream(cfg=cfg.TRAIN_DOWN, dataloader=loader, encoder=encoder, decoder=decoder)
        operator.train()


if __name__ == "__main__":
    main()
