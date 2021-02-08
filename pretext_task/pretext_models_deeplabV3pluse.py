"""
This script contains models of deeplab V3+ from paper:
    Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
"""
import torch
import torch.nn.functional as F
from torch import nn


class DepthwiseConv(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3, stride=1, dilation=1, bias=False):
        super(DepthwiseConv, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channels=in_chl, out_channels=in_chl,
                                            stride=stride, kernel_size=kernel_size,
                                            groups=in_chl, dilation=dilation,
                                            padding=dilation, bias=bias),
                                  nn.Conv2d(in_channels=in_chl, out_channels=out_chl, kernel_size=1, bias=bias),
                                  nn.BatchNorm2d(out_chl),
                                  nn.ReLU(inplace=False))

    def forward(self, x):
        out = self.body(x)

        return out


class Block(nn.Module):
    def __init__(self, in_chl, out_chl, exit=False, stride=2, dilation=1):
        super(Block, self).__init__()
        if exit:
            self.trunk = nn.Sequential(DepthwiseConv(in_chl, in_chl, dilation=dilation),
                                       DepthwiseConv(in_chl, out_chl, dilation=dilation),
                                       DepthwiseConv(out_chl, out_chl, dilation=dilation, stride=stride))
        else:
            self.trunk = nn.Sequential(DepthwiseConv(in_chl, out_chl, dilation=dilation),
                                       DepthwiseConv(out_chl, out_chl, dilation=dilation),
                                       DepthwiseConv(out_chl, out_chl, dilation=dilation, stride=stride))

        if in_chl == out_chl:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv2d(in_chl, out_chl, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = x
        out = self.trunk(x)
        if self.shortcut is not None:
            identity = self.shortcut(x)
        out += identity

        return out


class EntryFlow(nn.Module):
    def __init__(self):
        super(EntryFlow, self).__init__()
        self.block1 = Block(64, 128)
        self.block2 = Block(128, 256)
        self.block3 = Block(256, 728)

    def forward(self, x):
        out_block1 = self.block1(x)
        out = self.block2(out_block1)
        out = self.block3(out)

        return out_block1, out


class MiddleFlow(nn.Module):
    def __init__(self, times=16):
        super(MiddleFlow, self).__init__()
        self.blocks = [Block(728, 728, stride=1) for _ in range(times)]
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ExitFlow(nn.Module):
    def __init__(self, stride=2, dilation=2):
        super(ExitFlow, self).__init__()
        self.block = Block(728, 1024, stride=stride, exit=True, dilation=dilation)
        self.dw1 = DepthwiseConv(1024, 1536, dilation=dilation)
        self.dw2 = DepthwiseConv(1536, 1536, dilation=dilation)
        self.dw3 = DepthwiseConv(1536, 2048, dilation=dilation)

    def forward(self, x):
        out = self.block(x)
        out = self.dw1(out)
        out = self.dw2(out)
        out = self.dw3(out)

        return out


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.out = nn.Sequential(nn.AdaptiveAvgPool2d(2),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=False))

    def forward(self, x):
        size = x.shape[-2:]
        x = self.out(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_chl, out_chl, dilation_rate=None):
        super(ASPP, self).__init__()
        if dilation_rate is None:
            dilation_rate = [6, 12, 18]
        assert isinstance(dilation_rate, list)
        conv_list = []
        conv_list.append(nn.Sequential(nn.Conv2d(in_chl, out_chl, kernel_size=1, bias=False),
                                       nn.BatchNorm2d(out_chl),
                                       nn.ReLU(inplace=False)))
        conv_list.append(DepthwiseConv(in_chl, out_chl, dilation=dilation_rate[0]))
        conv_list.append(DepthwiseConv(in_chl, out_chl, dilation=dilation_rate[1]))
        conv_list.append(DepthwiseConv(in_chl, out_chl, dilation=dilation_rate[2]))
        conv_list.append(ASPPPooling(in_chl, out_chl))
        self.conv_list = nn.ModuleList(conv_list)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_chl, out_chl, 1, bias=False),
            nn.BatchNorm2d(out_chl),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1))

    def forward(self, x):
        res = []
        for conv in self.conv_list:
            res.append(conv(x))
        out = torch.cat(res, dim=1)

        return self.project(out)


class DeepLabV3PlusEncoder(nn.Module):
    def __init__(self, middle_times, exit_strid, exit_dilation):
        super(DeepLabV3PlusEncoder, self).__init__()
        self.in_convs = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=False),
                                      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=False))
        self.entry = EntryFlow()
        self.middle = MiddleFlow(times=middle_times)
        self.exit = ExitFlow(stride=exit_strid, dilation=exit_dilation)
        self.aspp = ASPP(in_chl=2048, out_chl=256)

    def forward(self, x):
        out = self.in_convs(x)
        out_middle_feature, entry_out = self.entry(out)
        middle_out = self.middle(entry_out)
        exit_out = self.exit(middle_out)
        aspp_out = self.aspp(exit_out)

        return out_middle_feature, aspp_out


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, out_fea):
        super(DeepLabV3PlusDecoder, self).__init__()
        self.in_conv_low = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU(inplace=False))
        self.out_conv = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=False),
                                      nn.Conv2d(256, out_fea, kernel_size=1, bias=False))

    def forward(self, low, aspp_out):
        out = torch.cat(
            [self.in_conv_low(low),
             nn.functional.interpolate(aspp_out, scale_factor=4, mode="bilinear", align_corners=True)], dim=1)
        out = self.out_conv(out)
        out = nn.functional.interpolate(out, scale_factor=4, mode="bilinear", align_corners=True)
        return out


class DecoderDeeplabBased(nn.Module):
    def __init__(self, out_fea):
        super(DecoderDeeplabBased, self).__init__()
        # self.decoder_depth = DeepLabV3PlusDecoder(out_fea["depth"])
        # self.decoder_instance = DeepLabV3PlusDecoder(out_fea['instance'])
        self.decoder_norm = DeepLabV3PlusDecoder(out_fea["norm"])
        self.decoder_edge = DeepLabV3PlusDecoder(out_fea["edge"])

    def forward(self, low, aspp_out):
        # out_depth = self.decoder_depth(low, aspp_out)
        # out_instance = self.decoder_instance(low, aspp_out)
        out_norm = self.decoder_norm(low, aspp_out)
        out_edge = self.decoder_edge(low, aspp_out)

        # return {"depth": out_depth, "normal": torch.sigmoid(out_norm), "edge": out_edge}
        # return {"instance": out_instance, "normal": torch.sigmoid(out_norm), "edge": out_edge}
        return {"normal": torch.sigmoid(out_norm), "edge": out_edge}
