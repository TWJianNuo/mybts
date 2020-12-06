# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch.nn.functional as F

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_input_channels=3):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def resnet_multiimage_input(num_layers, pretrained=False, num_input_channels=3):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_channels=num_input_channels)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])

        copys = int(num_input_channels / 3)
        conv1w = list()
        for i in range(copys):
            conv1w.append(loaded['conv1.weight'])
        conv1w.append(loaded['conv1.weight'][:, 0:int(num_input_channels-3*copys), :, :])
        loaded['conv1.weight'] = torch.cat(conv1w, dim=1) / num_input_channels * 3
        model.load_state_dict(loaded)
    return model

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_channels=3):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_channels != 3:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_channels)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = input_image
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))
        return self.features

class MultiModalityDecoder(nn.Module):
    def __init__(self, num_ch_enc, nchannelout=[1, 2], additionalblocks=2):
        super(MultiModalityDecoder, self).__init__()

        assert additionalblocks >= 0
        self.modalities = ['depth', 'shape']
        self.nchannelout = nchannelout
        self.upsample_mode = 'nearest'
        self.scales = range(4)
        self.additionalblocks = additionalblocks

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([24, 48, 96, 192, 384])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        # Four scales for depth
        for s in self.scales:
            for k in range(self.additionalblocks):
                self.convs[('depth', s, k)] = ConvBlock(self.num_ch_dec[s], self.num_ch_dec[s])
            self.convs[('depth', s)] = Conv3x3(self.num_ch_dec[s], self.nchannelout[0])

        # Single scale for shape
        for k in range(self.additionalblocks):
            self.convs[('shape', 0, k)] = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0])
        self.convs[('shape', 0)] = Conv3x3(self.num_ch_dec[0], self.nchannelout[1])

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            # Depth
            if i in self.scales:
                xmodality = x.clone()
                for k in range(self.additionalblocks):
                    xmodality = self.convs[('depth', i, k)](xmodality)
                self.outputs[('depth', i)] = self.sigmoid(self.convs[('depth', i)](xmodality))

            # Shape
            if i == 0:
                xmodality = x.clone()
                for k in range(self.additionalblocks):
                    xmodality = self.convs[('shape', i, k)](xmodality)
                self.outputs[('shape', i)] = self.sigmoid(self.convs[('shape', i)](xmodality))

        return self.outputs

class ShapeNet(nn.Module):
    def __init__(self, args):
        super(ShapeNet, self).__init__()
        self.args = args
        self.encoder = ResnetEncoder(50, pretrained=True)
        self.decoder = MultiModalityDecoder(self.encoder.num_ch_enc, nchannelout=[1, 2], additionalblocks=2)

    def forward(self, x):
        outputs = self.decoder(self.encoder(x))

        for key in outputs.keys():
            if 'depth' in key:
                outputs[key] = self.args.min_depth + outputs[key] * (self.args.max_depth - self.args.min_depth)
            elif 'shape' in key:
                outputs[key] = (outputs[key] - 0.5) * 2 * np.pi
            else:
                raise Exception("Specified modality is not initiated")
        return outputs