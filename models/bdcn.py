import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def crop(data1, data2, offset_h=0, offset_w=0):
    h1, w1 = data1.size()[2:]
    h2, w2 = data2.size()[2:]
    if h1 < h2 or w1 < w2:
        data1 = F.interpolate(data1, size=(h2, w2), mode='bilinear', align_corners=False)
        return data1
    sh = (h1 - h2) // 2 + offset_h
    sw = (w1 - w2) // 2 + offset_w
    return data1[:, :, sh:sh + h2, sw:sw + w2]


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class MSBlock(nn.Module):
    def __init__(self, c_in, rate=4):
        super(MSBlock, self).__init__()
        c_out = c_in
        self.rate = rate
        self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        dilation = self.rate*1 if self.rate >= 1 else 1
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        dilation = self.rate*2 if self.rate >= 1 else 1
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu2 = nn.ReLU(inplace=True)
        dilation = self.rate*3 if self.rate >= 1 else 1
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu3 = nn.ReLU(inplace=True)
        self._initialize_weights()

    def forward(self, x):
        o = self.relu(self.conv(x))
        o1 = self.relu1(self.conv1(o))
        o2 = self.relu2(self.conv2(o))
        o3 = self.relu3(self.conv3(o))
        out = o + o1 + o2 + o3
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

from torchvision.models import vgg16

class VGG16_C(nn.Module):
    def __init__(self, pretrain=True, logger=None):
        super(VGG16_C, self).__init__()
        vgg = vgg16(pretrained=pretrain).features

        self.conv1_1 = vgg[0]
        self.conv1_2 = vgg[2]
        self.conv2_1 = vgg[5]
        self.conv2_2 = vgg[7]
        self.conv3_1 = vgg[10]
        self.conv3_2 = vgg[12]
        self.conv3_3 = vgg[14]
        self.conv4_1 = vgg[17]
        self.conv4_2 = vgg[19]
        self.conv4_3 = vgg[21]
        self.conv5_1 = vgg[24]
        self.conv5_2 = vgg[26]
        self.conv5_3 = vgg[28]

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x):
        out1 = self.relu(self.conv1_1(x))
        out1 = self.relu(self.conv1_2(out1))
        out1p = self.maxpool(out1)

        out2 = self.relu(self.conv2_1(out1p))
        out2 = self.relu(self.conv2_2(out2))
        out2p = self.maxpool(out2)

        out3 = self.relu(self.conv3_1(out2p))
        out3 = self.relu(self.conv3_2(out3))
        out3 = self.relu(self.conv3_3(out3))
        out3p = self.maxpool(out3)

        out4 = self.relu(self.conv4_1(out3p))
        out4 = self.relu(self.conv4_2(out4))
        out4 = self.relu(self.conv4_3(out4))
        out4p = self.maxpool(out4)

        out5 = self.relu(self.conv5_1(out4p))
        out5 = self.relu(self.conv5_2(out5))
        out5 = self.relu(self.conv5_3(out5))

        return [
            out1, out1,
            out2, out2,
            out3, out3, out3,
            out4, out4, out4,
            out5, out5, out5
        ]

class BDCN(nn.Module):
    def __init__(self, pretrain=None, logger=None, rate=4):
        super(BDCN, self).__init__()
        self.pretrain = pretrain
        t = 1
        self.features = VGG16_C(pretrain, logger)
        self.msblock1_1 = MSBlock(64, rate)
        self.msblock1_2 = MSBlock(64, rate)
        self.conv1_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv1_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn1_1 = nn.Conv2d(21, 1, 1, stride=1)
        self.msblock2_1 = MSBlock(128, rate)
        self.msblock2_2 = MSBlock(128, rate)
        self.conv2_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv2_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn2 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn2_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.msblock3_1 = MSBlock(256, rate)
        self.msblock3_2 = MSBlock(256, rate)
        self.msblock3_3 = MSBlock(256, rate)
        self.conv3_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_3_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn3 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn3_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.msblock4_1 = MSBlock(512, rate)
        self.msblock4_2 = MSBlock(512, rate)
        self.msblock4_3 = MSBlock(512, rate)
        self.conv4_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv4_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv4_3_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn4 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn4_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.msblock5_1 = MSBlock(512, rate)
        self.msblock5_2 = MSBlock(512, rate)
        self.msblock5_3 = MSBlock(512, rate)
        self.conv5_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv5_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv5_3_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn5 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn5_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.upsample_2 = nn.ConvTranspose2d(1, 1, 4, stride=2, bias=False)
        self.upsample_4 = nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
        self.upsample_8 = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.upsample_8_5 = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.fuse = nn.Conv2d(10, 1, 1, stride=1)
        self._initialize_weights(logger)

    def forward(self, x):
        features = self.features(x)
        sum1 = self.conv1_1_down(self.msblock1_1(features[0])) + \
                self.conv1_2_down(self.msblock1_2(features[1]))
        s1 = self.score_dsn1(sum1)
        s11 = self.score_dsn1_1(sum1)
        sum2 = self.conv2_1_down(self.msblock2_1(features[2])) + \
            self.conv2_2_down(self.msblock2_2(features[3]))
        s2 = self.score_dsn2(sum2)
        s21 = self.score_dsn2_1(sum2)
        s2 = self.upsample_2(s2)
        s21 = self.upsample_2(s21)
        s2 = crop(s2, x, 1, 1)
        s21 = crop(s21, x, 1, 1)
        sum3 = self.conv3_1_down(self.msblock3_1(features[4])) + \
            self.conv3_2_down(self.msblock3_2(features[5])) + \
            self.conv3_3_down(self.msblock3_3(features[6]))
        s3 = self.score_dsn3(sum3)
        s3 =self.upsample_4(s3)
        s3 = crop(s3, x, 2, 2)
        s31 = self.score_dsn3_1(sum3)
        s31 =self.upsample_4(s31)
        s31 = crop(s31, x, 2, 2)
        sum4 = self.conv4_1_down(self.msblock4_1(features[7])) + \
            self.conv4_2_down(self.msblock4_2(features[8])) + \
            self.conv4_3_down(self.msblock4_3(features[9]))
        s4 = self.score_dsn4(sum4)
        s4 = self.upsample_8(s4)
        s4 = crop(s4, x, 4, 4)
        s41 = self.score_dsn4_1(sum4)
        s41 = self.upsample_8(s41)
        s41 = crop(s41, x, 4, 4)
        sum5 = self.conv5_1_down(self.msblock5_1(features[10])) + \
            self.conv5_2_down(self.msblock5_2(features[11])) + \
            self.conv5_3_down(self.msblock5_3(features[12]))
        s5 = self.score_dsn5(sum5)
        s5 = self.upsample_8_5(s5)
        s5 = crop(s5, x, 0, 0)
        s51 = self.score_dsn5_1(sum5)
        s51 = self.upsample_8_5(s51)
        s51 = crop(s51, x, 0, 0)
        o1, o2, o3, o4, o5 = s1.detach(), s2.detach(), s3.detach(), s4.detach(), s5.detach()
        o11, o21, o31, o41, o51 = s11.detach(), s21.detach(), s31.detach(), s41.detach(), s51.detach()
        p1_1 = s1
        p2_1 = s2 + o1
        p3_1 = s3 + o2 + o1
        p4_1 = s4 + o3 + o2 + o1
        p5_1 = s5 + o4 + o3 + o2 + o1
        p1_2 = s11 + o21 + o31 + o41 + o51
        p2_2 = s21 + o31 + o41 + o51
        p3_2 = s31 + o41 + o51
        p4_2 = s41 + o51
        p5_2 = s51

        fuse = self.fuse(torch.cat([p1_1, p2_1, p3_1, p4_1, p5_1, p1_2, p2_2, p3_2, p4_2, p5_2], 1))

        return [p1_1, p2_1, p3_1, p4_1, p5_1, p1_2, p2_2, p3_2, p4_2, p5_2, fuse]

    def _initialize_weights(self, logger=None):
        for name, param in self.state_dict().items():
            if self.pretrain and 'features' in name:
                continue
            elif 'upsample' in name:
                if logger:
                    logger.info('init upsamle layer %s ' % name)
                k = int(name.split('.')[0].split('_')[1])
                param.copy_(get_upsampling_weight(1, 1, k*2))
            elif 'fuse' in name:
                if logger:
                    logger.info('init params %s ' % name)
                if 'bias' in name:
                    param.zero_()
                else:
                    nn.init.constant_(param, 0.080)
            else:
                if logger:
                    logger.info('init params %s ' % name)
                if 'bias' in name:
                    param.zero_()
                else:
                    param.normal_(0, 0.01)

if __name__ == '__main__':
    model = BDCN(pretrain=True)
    a = torch.rand((2, 3, 100, 100))
    for x in model(a):
        print(x.shape)
