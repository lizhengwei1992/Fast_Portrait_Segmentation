'''
	mobilenetv2_dilate_unet

Author: Zhengwei Li
Data: July 20 2018
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

#-------------------------------------------------------------------------------------------------
# MobileNet_v2_os_32_MFo
#--------------------
class MobileNet_v2_os_32_MFo(nn.Module):
    def __init__(self, nInputChannels=3):
        super(MobileNet_v2_os_32_MFo, self).__init__()
        # 1/2
        # 256 x 256
        self.head_conv = conv_bn(nInputChannels, 32, 2)

        # 1/2
        # 256 x 256
        self.block_1 = InvertedResidual(32, 16, 1, 1)
        # 1/4 128 x 128
        self.block_2 = nn.Sequential( 
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6)
            )
        # 1/8 64 x 64
        self.block_3 = nn.Sequential( 
            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6)
            )
        # 1/16 32 x 32
        self.block_4 = nn.Sequential( 
            InvertedResidual(32, 64, 2, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6)            
            )
        # 1/16 32 x 32
        self.block_5 = nn.Sequential( 
            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6)          
            )
        # 1/32 16 x 16
        self.block_6 = nn.Sequential( 
            InvertedResidual(96, 160, 2, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6)          
            )
        # 1/32 16 x 16
        self.block_7 = InvertedResidual(160, 320, 1, 6)


    def forward(self, x):
        x = self.head_conv(x)

        x1 = self.block_1(x)
        x2 = self.block_2(x1)
        x3 = self.block_3(x2)
        x4 = self.block_4(x3)
        x4 = self.block_5(x4)
        x5 = self.block_6(x4)
        x5 = self.block_7(x5)

        return x1, x2, x3, x4, x5



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
# mv2_dilate_unet
# feature exstractor : MobileNet_v2_os_32_MFo
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
