import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import Res2Net as Pre_Res2Net
import os
import common
from lka import AttentionModule
# from retinex import retinex_tensor

class sub_pixel(nn.Module):
    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()
        modules = []
        modules.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*modules)
    def forward(self, x):
        x = self.body(x)
        return x
from rcan import rcan


###################################################################################################################
class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):

        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x = self.layer3(x_layer2)  # x16

        return x, x_layer1, x_layer2


######################
# decoder
######################
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class DehazeBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(DehazeBlock, self).__init__()

        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x

        return res


class Enhancer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Enhancer, self).__init__()

        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.tanh = nn.Tanh()

        self.refine1 = nn.Conv2d(in_channels, 20, kernel_size=3, stride=1, padding=1)
        self.refine2 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1)

        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)

        self.refine3 = nn.Conv2d(20 + 4, out_channels, kernel_size=3, stride=1, padding=1)
        self.upsample = F.upsample_nearest

        self.batch1 = nn.InstanceNorm2d(100, affine=True)

    def forward(self, x):
        dehaze = self.relu((self.refine1(x)))
        dehaze = self.relu((self.refine2(dehaze)))
        shape_out = dehaze.data.size()

        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(dehaze, 32)

        x102 = F.avg_pool2d(dehaze, 16)

        x103 = F.avg_pool2d(dehaze, 8)

        x104 = F.avg_pool2d(dehaze, 4)

        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1)
        dehaze = self.tanh(self.refine3(dehaze))

        return dehaze


class Dehaze(nn.Module):
    def __init__(self, imagenet_model):
        super(Dehaze, self).__init__()

        self.encoder = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4)
        res2net50 = Pre_Res2Net.Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4)
        pretrained_dict = {}
        weight_names = [
            'res2net50_v1b_26w_4s-06e79181.pth',
            'res2net50_v1b_26w_4s-3cf99910.pth',
        ]
        for w in weight_names:
            weight_path = os.path.join(imagenet_model, w)
            if not os.path.isfile(weight_path):
                continue
            try:
                state = torch.load(weight_path, map_location='cpu')
                res2net50.load_state_dict(state)
                pretrained_dict = res2net50.state_dict()
                break
            except Exception:
                continue
        if pretrained_dict:
            model_dict = self.encoder.state_dict()
            key_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(key_dict)
            self.encoder.load_state_dict(model_dict)
        mid_channels = 512
        mid_inner = 320
        self.mid_reduce = nn.Conv2d(1024, mid_inner, kernel_size=1, stride=1, padding=0)
        self.mid_conv = DehazeBlock(default_conv, mid_inner, 3)
        self.mid_expand = nn.Conv2d(mid_inner, mid_channels, kernel_size=1, stride=1, padding=0)

        self.up_block1 = nn.PixelShuffle(2)
        self.attention1 = DehazeBlock(default_conv, 128, 3)
        self.attention2 = DehazeBlock(default_conv, 160, 3)
        self.lka1 = AttentionModule(128)
        self.lka2 = AttentionModule(160)
        self.lka3 = AttentionModule(104)
        self.enhancer = Enhancer(26, 26)


    def forward(self, input):
        x, x_layer1, x_layer2 = self.encoder(input)

        x_mid = self.mid_reduce(x)
        x_mid = self.mid_conv(x_mid)
        x_mid = self.mid_expand(x_mid)

        x = self.up_block1(x_mid)
        x = self.attention1(x)
        x = self.lka1(x)

        if x.size()[2:] != x_layer2.size()[2:]:
             x = F.interpolate(x, size=x_layer2.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x, x_layer2), 1)
        x = self.up_block1(x)
        x = self.attention2(x)
        x = self.lka2(x)

        if x.size()[2:] != x_layer1.size()[2:]:
             x = F.interpolate(x, size=x_layer1.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x, x_layer1), 1)
        x = self.up_block1(x)
        x = self.lka3(x)
        x = self.up_block1(x)

        # x = self.lka4(x)

        dout2 = self.enhancer(x)
        #torch.Size([2, 28, 256, 256])

        return dout2

class fusion_refine(nn.Module):
    def __init__(self, imagenet_model, rcan_model):
        super(fusion_refine, self).__init__()
        self.feature_extract = Dehaze(imagenet_model)
        self.pre_trained_rcan = rcan()
        self.tail1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(58, 3, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, input):
        feature = self.feature_extract(input)
        rcan_feat, rcan_reflect = self.pre_trained_rcan(input)
        if rcan_feat.size()[2:] != feature.size()[2:]:
            rcan_feat = F.interpolate(
                rcan_feat,
                size=feature.size()[2:],
                mode='bilinear',
                align_corners=True
            )
        if rcan_reflect.size()[2:] != input.size()[2:]:
            rcan_reflect = F.interpolate(
                rcan_reflect,
                size=input.size()[2:],
                mode='bilinear',
                align_corners=True
            )
        x = torch.cat([feature, rcan_feat], 1)
        feat_hazy = self.tail1(x)
        if feat_hazy.size()[2:] != input.size()[2:]:
            feat_hazy = F.interpolate(feat_hazy, size=input.size()[2:], mode='bilinear', align_corners=True)
        return feat_hazy, rcan_reflect
