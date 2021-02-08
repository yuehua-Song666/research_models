import torch
from torch.utils.data import DataLoader
from operate import OperatorDownstream
from configs import cfg
from pretext_task.pretext_models_FPN import FPNBasedEncoder
from downstream_task.downstream_models import DownstreamLinear
from downstream_task.dataset import DatasetDownstreamValADE


def main(cfg):
    # load encoder, decoder
    encoder = FPNBasedEncoder("resnet50")
    encoder.load_state_dict(torch.load(cfg.EVAL.encoder))
    decoder = DownstreamLinear(num_class=cfg.DATASET_DOWN.num_class)
    decoder.load_state_dict(torch.load(cfg.EVAL.decoder))
    # load dataset
    dataset_val = DatasetDownstreamValADE(cfg.EVAL.list_eval, cfg.EVAL)
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, drop_last=True)
    # load operator
    operator = OperatorDownstream(cfg.EVAL, encoder, decoder, dataloader_val)
    operator.eval()


if __name__ == "__main__":
    main(cfg)
