# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

import torch
import torch.nn as nn
import torch.nn.functional as torch_nn_func
import torch.distributed as dist
import numpy as np
# This sets the batch norm layers in pytorch as if {'is_training': False, 'scale': True} in tensorflow
def bn_init_as_tf(m):
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = True  # These two lines enable using stats (moving mean and var) loaded from pretrained model
        m.eval()  # or zero mean and variance of one if the batch norm layer has no pretrained values
        m.affine = True
        m.requires_grad = True

def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        if m.weight.requires_grad == True:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

class atrous_conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, apply_bn_first=True):
        super(atrous_conv, self).__init__()
        self.atrous_conv = torch.nn.Sequential()
        if apply_bn_first:
            self.atrous_conv.add_module('first_bn', nn.BatchNorm2d(in_channels, momentum=0.01, affine=True, track_running_stats=True, eps=1.1e-5))
        self.atrous_conv.add_module('aconv_sequence', nn.Sequential(nn.ReLU(),
                                                                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels * 2, bias=False, kernel_size=1, stride=1, padding=0),
                                                                    nn.BatchNorm2d(out_channels * 2, momentum=0.01, affine=True, track_running_stats=True),
                                                                    nn.ReLU(),
                                                                    nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, bias=False, kernel_size=3, stride=1, padding=(dilation, dilation), dilation=dilation)))
    def forward(self, x):
        return self.atrous_conv.forward(x)

class upconv(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2):
        super(upconv, self).__init__()
        self.elu = nn.ELU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=3, stride=1, padding=1)
        self.ratio = ratio

    def forward(self, x):
        up_x = torch_nn_func.interpolate(x, scale_factor=self.ratio, mode='nearest')
        out = self.conv(up_x)
        out = self.elu(out)
        return out

class reduction_1x1(nn.Sequential):
    def __init__(self, num_in_filters, num_out_filters):
        super(reduction_1x1, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.reduc = torch.nn.Sequential()

        while num_out_filters >= 4:
            if num_out_filters < 8:
                self.reduc.add_module('final', torch.nn.Sequential(nn.Conv2d(num_in_filters, out_channels=2, bias=False, kernel_size=1, stride=1, padding=0), nn.Sigmoid()))
                break
            else:
                self.reduc.add_module('inter_{}_{}'.format(num_in_filters, num_out_filters), torch.nn.Sequential(nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters, bias=False, kernel_size=1, stride=1, padding=0), nn.ELU()))
            num_in_filters = num_out_filters
            num_out_filters = num_out_filters // 2

    def forward(self, netinput):
        activation = self.reduc.forward(netinput)
        return activation

class ShapenetDecoder(nn.Module):
    def __init__(self, params, feat_out_channels, num_features=512):
        super(ShapenetDecoder, self).__init__()
        self.params = params

        if dist.get_world_size() == 1:
            BNlayer = nn.BatchNorm2d
            print("Init Batch Norm layer as nn.BatchNorm2d")
        else:
            BNlayer = nn.SyncBatchNorm
            print("Init Batch Norm layer as nn.SyncBatchNorm")

        self.upconv5 = upconv(feat_out_channels[4], num_features)
        self.bn5 = BNlayer(num_features, momentum=0.01, affine=True, eps=1.1e-5)

        self.conv5 = torch.nn.Sequential(nn.Conv2d(num_features + feat_out_channels[3], num_features, 3, 1, 1, bias=False), nn.ELU())
        self.upconv4 = upconv(num_features, num_features // 2)
        self.bn4 = BNlayer(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv4 = torch.nn.Sequential(nn.Conv2d(num_features // 2 + feat_out_channels[2], num_features // 2, 3, 1, 1, bias=False), nn.ELU())
        self.bn4_2 = BNlayer(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)

        self.daspp_3 = atrous_conv(num_features // 2, num_features // 4, 3, apply_bn_first=False)
        self.daspp_6 = atrous_conv(num_features // 2 + num_features // 4 + feat_out_channels[2], num_features // 4, 6)
        self.daspp_12 = atrous_conv(num_features + feat_out_channels[2], num_features // 4, 12)
        self.daspp_18 = atrous_conv(num_features + num_features // 4 + feat_out_channels[2], num_features // 4, 18)
        self.daspp_24 = atrous_conv(num_features + num_features // 2 + feat_out_channels[2], num_features // 4, 24)
        self.daspp_conv = torch.nn.Sequential(nn.Conv2d(num_features + num_features // 2 + num_features // 4, num_features // 4, 3, 1, 1, bias=False), nn.ELU())
        self.reduc8x8 = reduction_1x1(num_features // 4, num_features // 4)

        self.upconv3 = upconv(num_features // 4, num_features // 4)
        self.bn3 = BNlayer(num_features // 4, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv3 = torch.nn.Sequential(nn.Conv2d(num_features // 4 + feat_out_channels[1] + 2, num_features // 4, 3, 1, 1, bias=False), nn.ELU())
        self.reduc4x4 = reduction_1x1(num_features // 4, num_features // 8)

        self.upconv2 = upconv(num_features // 4, num_features // 8)
        self.bn2 = BNlayer(num_features // 8, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv2 = torch.nn.Sequential(nn.Conv2d(num_features // 8 + feat_out_channels[0] + 2, num_features // 8, 3, 1, 1, bias=False), nn.ELU())

        self.reduc2x2 = reduction_1x1(num_features // 8, num_features // 16)

        self.upconv1 = upconv(num_features // 8, num_features // 16)
        self.reduc1x1 = reduction_1x1(num_features // 16, num_features // 32)
        self.conv1 = torch.nn.Sequential(nn.Conv2d(num_features // 16 + 8, num_features // 16, 3, 1, 1, bias=False), nn.ELU())
        self.get_shape = torch.nn.Sequential(nn.Conv2d(num_features // 16, 2, 3, 1, 1, bias=False), nn.Sigmoid())

    def forward(self, features):
        skip0, skip1, skip2, skip3 = features[1], features[2], features[3], features[4]
        dense_features = torch.nn.ReLU()(features[5])
        upconv5 = self.upconv5(dense_features)  # H/16
        upconv5 = self.bn5(upconv5)
        concat5 = torch.cat([upconv5, skip3], dim=1)
        iconv5 = self.conv5(concat5)

        upconv4 = self.upconv4(iconv5)  # H/8
        upconv4 = self.bn4(upconv4)
        concat4 = torch.cat([upconv4, skip2], dim=1)
        iconv4 = self.conv4(concat4)
        iconv4 = self.bn4_2(iconv4)

        daspp_3 = self.daspp_3(iconv4)
        concat4_2 = torch.cat([concat4, daspp_3], dim=1)
        daspp_6 = self.daspp_6(concat4_2)
        concat4_3 = torch.cat([concat4_2, daspp_6], dim=1)
        daspp_12 = self.daspp_12(concat4_3)
        concat4_4 = torch.cat([concat4_3, daspp_12], dim=1)
        daspp_18 = self.daspp_18(concat4_4)
        concat4_5 = torch.cat([concat4_4, daspp_18], dim=1)
        daspp_24 = self.daspp_24(concat4_5)
        concat4_daspp = torch.cat([iconv4, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24], dim=1)
        daspp_feat = self.daspp_conv(concat4_daspp)

        reduc8x8 = self.reduc8x8(daspp_feat)
        reduc8x8_scaled = torch_nn_func.interpolate(reduc8x8, scale_factor=8, mode='nearest')

        upconv3 = self.upconv3(daspp_feat)  # H/4
        upconv3 = self.bn3(upconv3)
        concat3 = torch.cat([upconv3, skip1, torch_nn_func.interpolate(reduc8x8, scale_factor=2, mode='nearest')], dim=1)
        iconv3 = self.conv3(concat3)

        reduc4x4 = self.reduc4x4(iconv3)
        reduc4x4_scaled = torch_nn_func.interpolate(reduc4x4, scale_factor=4, mode='nearest')

        upconv2 = self.upconv2(iconv3)  # H/2
        upconv2 = self.bn2(upconv2)
        concat2 = torch.cat([upconv2, skip0, torch_nn_func.interpolate(reduc4x4, scale_factor=2, mode='nearest')], dim=1)
        iconv2 = self.conv2(concat2)

        reduc2x2 = self.reduc2x2(iconv2)
        reduc2x2_scaled = torch_nn_func.interpolate(reduc2x2, scale_factor=2, mode='nearest')

        upconv1 = self.upconv1(iconv2)
        reduc1x1 = self.reduc1x1(upconv1)
        concat1 = torch.cat([upconv1, reduc1x1, reduc2x2_scaled, reduc4x4_scaled, reduc8x8_scaled], dim=1)
        iconv1 = self.conv1(concat1)
        shape = (self.get_shape(iconv1) - 0.5) * np.pi * 2

        return reduc8x8_scaled, reduc4x4_scaled, reduc2x2_scaled, reduc1x1, shape

class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        import torchvision.models as models
        if params.encoder == 'densenet121_bts':
            self.base_model = models.densenet121(pretrained=True).features
            self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
            self.feat_out_channels = [64, 64, 128, 256, 1024]
        elif params.encoder == 'densenet161_bts':
            self.base_model = models.densenet161(pretrained=True).features
            self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
            self.feat_out_channels = [96, 96, 192, 384, 2208]
        elif params.encoder == 'resnet50_bts':
            self.base_model = models.resnet50(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'resnet101_bts':
            self.base_model = models.resnet101(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'resnext50_bts':
            self.base_model = models.resnext50_32x4d(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'resnext101_bts':
            self.base_model = models.resnext101_32x8d(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        else:
            print('Not supported encoder: {}'.format(params.encoder))

    def forward(self, x):
        features = [x]
        skip_feat = [x]
        for k, v in self.base_model._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            feature = v(features[-1])
            features.append(feature)
            if any(x in k for x in self.feat_names):
                skip_feat.append(feature)

        return skip_feat

class ShapeNet(nn.Module):
    def __init__(self, params):
        super(ShapeNet, self).__init__()
        self.encoder = Encoder(params)
        self.decoder = ShapenetDecoder(params, self.encoder.feat_out_channels, params.bts_size)

    def forward(self, x):
        skip_feat = self.encoder(x)
        return self.decoder(skip_feat)