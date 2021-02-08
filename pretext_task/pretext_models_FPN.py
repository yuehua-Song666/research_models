import torch
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torchvision.models import resnet
from torch import nn
from .se_module import SEBottleneck

layerNums = {"resnet18": [2, 2, 2, 2], "resnet34": [3, 4, 6, 3], "resnet50": [3, 4, 6, 3],
             "resnet101": [3, 4, 23, 3], "resnet152": [[3, 8, 36, 3]]}

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = resnet.ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


class FPNBasedEncoder(nn.Module):
    """
    most part of codes from https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py
    :param model: select the resnet version
    :param pretrain: pretrained model?
    """

    def __init__(self, model, se=False, pretrain=False):
        super(FPNBasedEncoder, self).__init__()
        if se:
            bottleneck = SEBottleneck
        else:
            bottleneck = resnet.Bottleneck
        baseModel = _resnet(model, bottleneck, layerNums[model], pretrain, progress=False)
        self.head = nn.Sequential(
            baseModel.conv1,
            baseModel.bn1,
            baseModel.relu,
            baseModel.maxpool,
        )

        # bottom up layers
        self.layer1 = baseModel.layer1
        self.layer2 = baseModel.layer2
        self.layer3 = baseModel.layer3
        self.layer4 = baseModel.layer4
        # top layer
        self.top = nn.Sequential(nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0),
                                 nn.BatchNorm2d(256),
                                 nn.Tanh())

    def forward(self, x):
        # bottom up
        c0 = self.head(x)
        c1 = self.layer1(c0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        # top down
        # p5 output(n, 256, 8, 8) for image size (256, 256)
        top = self.top(c4)
        # todo: change the format on training script and model structure of decoder
        return [c1, c2, c3, c4, top]


class FPNBasedDecoder(nn.Module):
    def __init__(self):
        super(FPNBasedDecoder, self).__init__()
        # Smooth layers
        self.smooth_norm = nn.Sequential(nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1),
                                         nn.Sigmoid())
        self.smooth_depth = nn.Sequential(nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(inplace=True))
        self.smooth_edge = nn.Sequential(nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1))

        self.deconv_norm1 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, dilation=1, padding=1),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(True))
        self.deconv_edge1 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, dilation=1, padding=1),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(True))
        self.deconv_depth1 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, dilation=1, padding=1),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU(True))

        self.deconv_norm2 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, dilation=1, padding=1),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(True))
        self.deconv_edge2 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, dilation=1, padding=1),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(True))
        self.deconv_depth2 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, dilation=1, padding=1),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU(True))
        # Lateral layers
        self.latlayer00 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer01 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer02 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayers = [self.latlayer00, self.latlayer01, self.latlayer02]

        # conv layers
        self.conv00 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))
        self.conv01 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))

        self.convs = [self.conv00, self.conv01]

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def upsample_cat(self, x, y):
        """

        :param x: feature to be upsampled
        :param y: lateral features
        :return: added features
        """
        _, _, h, w = y.size()

        return torch.cat([F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True), y], dim=1)

    def _forward(self, inputs, lats, cons, top):
        if inputs:
            x = self.upsample_cat(top, lats.pop()(inputs.pop()))
            if cons:
                x = cons.pop()(x)

            return self._forward(inputs, lats, cons, x)
        else:
            return top

    def forward(self, x1, x2, x3, top):
        out = self._forward([x1, x2, x3], self.latlayers[::-1], self.convs[::-1], top)
        out_norm = self.deconv_norm1(out)
        out_norm = self.deconv_norm2(out_norm)
        out_norm = self.smooth_norm(out_norm)

        out_depth = self.deconv_depth1(out)
        out_depth = self.deconv_depth2(out_depth)
        out_depth = self.smooth_depth(out_depth)

        out_edge = self.deconv_edge1(out)
        out_edge = self.deconv_edge2(out_edge)
        out_edge = self.smooth_edge(out_edge)

        return {'normal': out_norm, 'depth': out_depth, 'edge': out_edge}


class Discriminator(nn.Module):
    """
    in: last feature from encoder
    """

    def __init__(self, ):
        super(Discriminator, self).__init__()
        # FPN: input size: batch x 256 x 8 x 8 for input size 256
        #           batch x 256 x 12 x 12 for input size 384
        # deeplabV3+: batch x 256 x 16 x 16 for input size 256
        self.conv1 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(1024), nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0),
                                   nn.Sigmoid())

    def forward(self, x):
        a = x.size()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x.view(-1)

class Discriminator_after_decoder(nn.Module):
    """
    in: last feature from encoder
    """

    def __init__(self, ):
        super(Discriminator_after_decoder, self).__init__()
        # FPN: input size: batch x 256 x 8 x 8 for input size 256
        #           batch x 256 x 12 x 12 for input size 384
        # deeplabV3+: batch x 256 x 16 x 16 for input size 256
        self.convEdge = nn.Sequential(nn.Conv2d(1, 3, kernel_size=5, stride=1, padding=2))
        self.conv1 = nn.Sequential(nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=2),
                                   nn.BatchNorm2d(8), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
                                   nn.BatchNorm2d(16), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
                                   nn.BatchNorm2d(32), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
                                   nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                                   nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                                   nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
                                   nn.Sigmoid())

    def forward(self, x, is_edge = False):
        if is_edge:
            x = self.convEdge(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        return x.view(-1)


class FPNBasedDecoderDilated(nn.Module):
    def __init__(self):
        super(FPNBasedDecoderDilated, self).__init__()
        # Smooth layers
        self.smooth_norm = nn.Sequential(nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=1),
                                         nn.Sigmoid())
        self.smooth_depth = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=1),
                                          nn.ReLU(inplace=True))
        self.smooth_edge = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=1))

        self.deconv_norm1 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, dilation=1, padding=1),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(True))
        self.deconv_edge1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, dilation=3, padding=3, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True))
        self.deconv_depth1 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, dilation=1, padding=1),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU(True))

        self.deconv_norm2 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, dilation=1, padding=1),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(True))
        self.deconv_edge2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, dilation=3, padding=3, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True))
        self.deconv_depth2 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, dilation=1, padding=1),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU(True))
        # Lateral layers
        self.latlayer00 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer01 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer02 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayers = [self.latlayer00, self.latlayer01, self.latlayer02]

        # conv layers
        self.conv00 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=3, dilation=3),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))
        self.conv01 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=3, dilation=3),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))

        self.convs = [self.conv00, self.conv01]

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def upsample_cat(self, x, y):
        """

        :param x: feature to be upsampled
        :param y: lateral features
        :return: added features
        """
        _, _, h, w = y.size()

        return torch.cat([F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True), y], dim=1)

    def _forward(self, inputs, lats, cons, top):
        if inputs:
            x = self.upsample_cat(top, lats.pop()(inputs.pop()))
            if cons:
                x = cons.pop()(x)

            return self._forward(inputs, lats, cons, x)
        else:
            return top

    def forward(self, x1, x2, x3, top):
        out = self._forward([x1, x2, x3], self.latlayers[::-1], self.convs[::-1], top)
        out_norm = self.deconv_norm1(out)
        out_norm = self.deconv_norm2(out_norm)
        out_norm = self.smooth_norm(out_norm)

        out_depth = self.deconv_depth1(out)
        out_depth = self.deconv_depth2(out_depth)
        out_depth = self.smooth_depth(out_depth)

        out_edge = self.deconv_edge1(out)
        out_edge = self.deconv_edge2(out_edge)
        out_edge = self.smooth_edge(out_edge)

        return {'normal': out_norm, 'depth': out_depth, 'edge': out_edge}


class FPNBasedDecoderInterpolate(nn.Module):
    def __init__(self):
        super(FPNBasedDecoderInterpolate, self).__init__()
        # Smooth layers
        self.smooth_norm = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout2d(0.1),
                                         nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=0),
                                         nn.Sigmoid())
        self.smooth_depth = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout2d(0.1),
                                          nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0),
                                          nn.ReLU(inplace=True))
        self.smooth_edge = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout2d(0.1),
                                         nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0))

        # Lateral layers
        self.latlayer00 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer01 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer02 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayers = [self.latlayer00, self.latlayer01, self.latlayer02]

        # conv layers
        self.conv00 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))
        self.conv01 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))
        self.conv02 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))

        self.convs = [self.conv00, self.conv01, self.conv02]

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def upsample_cat(self, x, y):
        """

        :param x: feature to be upsampled
        :param y: lateral features
        :return: added features
        """
        _, _, h, w = y.size()

        return torch.cat([F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True), y], dim=1)

    def _forward(self, inputs, lats, cons, top):
        if inputs:
            x = self.upsample_cat(top, lats.pop()(inputs.pop()))
            x = cons.pop()(x)

            return self._forward(inputs, lats, cons, x)
        else:
            return top

    def forward(self, x1, x2, x3, top):
        out = self._forward([x1, x2, x3], self.latlayers[::-1], self.convs[::-1], top)
        out = nn.functional.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)
        out_norm = self.smooth_norm(out)
        out_depth = self.smooth_depth(out)
        out_edge = self.smooth_edge(out)

        return {'normal': out_norm, 'depth': out_depth, 'edge': out_edge}
