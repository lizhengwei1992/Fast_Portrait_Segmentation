'''
	mobilenetv2_dilate_skip_net

Author: Zhengwei Li
Data: July 20 2018
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# mobilenet_v2
from model.MobileNet_v2 import MobileNet_v2_os_32_MFo
from model.layers import *
import pdb


INPUT_SIZE = 512


# up amd concat and dilate
class UCD(nn.Module):
    def __init__(self, inplanes, planes, dilation):
        super(UCD, self).__init__()


        self.up = nn.ConvTranspose2d(inplanes, planes, kernel_size=2, stride=2, padding=0)
        self.aspp = nn.Sequential(nn.Conv2d(planes*2, planes*2, kernel_size=3,
                                            stride=1, padding=dilation, dilation=dilation, bias=False),
                    nn.BatchNorm2d(planes*2))

    def forward(self, e, x):

        x = self.up(x)
        x = torch.cat((x, e), dim=1)
        x = self.aspp(x)

        return x

#-------------------------------------------------------------------------------------------------
# mv2_grade_dilate_Dnet
# feature exstractor : MobileNet_v2_os_MFo
#-----------------------------------------
class MobileNet_v2_Dilate_Unet(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=1):

        super(MobileNet_v2_Dilate_Unet, self).__init__()

        # mobilenetv2 feature 
        self.mobilenet_features = MobileNet_v2_os_32_MFo(nInputChannels)

        self.up_concat_dilate_1 = UCD(320, 96, dilation = 2)
        self.up_concat_dilate_2 = UCD(192, 32, dilation = 6)
        self.up_concat_dilate_3 = UCD(64, 24, dilation = 12)
        self.up_concat_dilate_4 = UCD(48, 16, dilation = 18)

        self.last_conv = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.Conv2d(32, n_classes, kernel_size=1, stride=1))
        # init weights
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # x1 - x4 : 1/8 64 x 64
        e1, e2, e3, e4, feature_map = self.mobilenet_features(x)


        feature_map = self.up_concat_dilate_1(e4, feature_map) 
        feature_map = self.up_concat_dilate_2(e3, feature_map) 
        feature_map = self.up_concat_dilate_3(e2, feature_map) 
        feature_map = self.up_concat_dilate_4(e1, feature_map) 

        heat_map = self.last_conv(feature_map) 

        heat_map = F.upsample(heat_map, scale_factor=2, mode='bilinear', align_corners=True)   


        return heat_map
