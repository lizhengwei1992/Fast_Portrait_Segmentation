import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn(inp, oup, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.PReLU(oup)
    )

class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(make_dense, self).__init__()

        self.bn = nn.BatchNorm2d(nChannels)
        self.act = nn.PReLU(nChannels)
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)
        

    def forward(self, input):
        
        x = self.bn(input)
        x = self.act(x)
        x = self.conv(x)

        out = torch.cat((input, x), 1)
        return out

class DenseBlock(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(DenseBlock, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):    
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate 
        self.dense_layers = nn.Sequential(*modules)


    def forward(self, x):
        out = self.dense_layers(x)

        return out

# DenseResidualBlock
class DRB(nn.Module):
    def __init__(self, nIn, s=4, add=True):

        super(DRB, self).__init__()

        n = int(nIn//s) 

        self.conv =  nn.Conv2d(nIn, n, 3, stride=1, padding=1, bias=False)

        self.bn = nn.BatchNorm2d(n)
        self.act = nn.PReLU(n)

        self.dense_block = DenseBlock(n, nDenselayer=(s-1), growthRate=n)

        self.add = add

    def forward(self, input):

        residual = input
        # reduce
        x = self.conv(input)
        x = self.bn(x)
        x = self.act(x)
        x =self.dense_block(x)

        if self.add:
            out = x + residual
        else:
            out = x

        return out

# Attention Refinement Module (ARM)
class ARM(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(ARM, self).__init__()

        self.global_pool = nn.AvgPool2d(kernel_size, stride=kernel_size)
        self.conv_1x1 = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False)
        # self.bn = nn.BatchNorm1d(in_channels)
        self.sigmod = nn.Sigmoid()

    def forward(self, input):
        x = self.global_pool(input)

        x = self.conv_1x1(x)
        # x = self.bn(x)

        x = self.sigmod(x)

        out = torch.mul(input, x)
        return out

# Feature Fusion Module
class FFM(nn.Module):
    def __init__(self, in_channels, kernel_size, alpha=3):
        super(FFM, self).__init__()
        inter_channels = in_channels // alpha
        self.conv_bn_relu = conv_bn(in_channels, inter_channels, kernel_size=1, padding=0)

        self.global_pool = nn.AvgPool2d(kernel_size, stride=kernel_size)
        self.conv_1x1_1 = nn.Conv2d(inter_channels, inter_channels, 1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()
        self.conv_1x1_2 = nn.Conv2d(inter_channels, inter_channels, 1, stride=1, padding=0, bias=False)
        self.sigmod = nn.Sigmoid()

        self.classifier = nn.Conv2d(inter_channels, 1, 1, stride=1, padding=0, bias=True)

    def forward(self, input):

        input = self.conv_bn_relu(input)

        x = self.global_pool(input)

        x = self.conv_1x1_1(x)
        x = self.relu(x)
        x = self.conv_1x1_2(x)
        x = self.sigmod(x)

        out = torch.mul(input, x)
        out = input + out

        out = self.classifier(out)
        return out
 # -----------------------------------------------------------------------------------
 # RD_BiSeNet sparsity-regularization
 # -----------------------------------------------------------------------------------
class RD_BiSeNet(nn.Module):

    def __init__(self, classes=1, cfg=None):

        super(RD_BiSeNet, self).__init__()


        # -----------------------------------------------------------------
        # Spatial Path 
        # ---------------------
        self.conv_bn_relu_1 = conv_bn(3, 8, stride=2)
        self.conv_bn_relu_2 = conv_bn(8, 12, stride=2)
        self.conv_bn_relu_3 = conv_bn(12, 16, stride=2)
        # -----------------------------------------------------------------
        # Context Path 
        # ---------------------
        self.conv = conv_bn(3, 8, stride=2)
        self.stage_0 = DRB(8, s=2, add=True)

        self.down_1 = conv_bn(8, 12, stride=2) 
        self.stage_1 = DRB(12, s=3, add=True)

        self.down_2 = conv_bn(12, 24, stride=2)        
        self.stage_2 = nn.Sequential(DRB(24, s=6, add=True),
                                     DRB(24, s=6, add=True))

        self.down_3 = conv_bn(24, 48, stride=2)  
        self.stage_3 = nn.Sequential(DRB(48, s=6, add=True),
                                     DRB(48, s=6, add=True))

        self.down_4 = conv_bn(48, 64, stride=2)  
        self.stage_4 = nn.Sequential(DRB(64, s=8, add=True),
                                     DRB(64, s=8, add=True))

        # ARM
        self.arm_16 = ARM(48, kernel_size=32)
        self.arm_32 = ARM(64, kernel_size=16)

        self.global_pool = nn.AvgPool2d(kernel_size=16, stride=16)
        self.tail_up = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.level_16_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.level_32_up = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                                         nn.Conv2d(64, 48, 1, stride=1, padding=0, bias=True))

        # FFM
        self.ffm = FFM(48+16, kernel_size=64, alpha=1)

        self.up = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

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

    def forward(self, input):

        # -----------------------------------------------
        # Spatial Path 
        # ---------------------
        spatial = self.conv_bn_relu_1(input)
        spatial = self.conv_bn_relu_2(spatial)
        spatial = self.conv_bn_relu_3(spatial)
        # -----------------------------------------------
        # Context Path 
        # ---------------------
        x = self.conv(input)
        s0 = self.stage_0(x)
        # 1/4
        s1_0 = self.down_1(s0)
        s1 = self.stage_1(s1_0)
        # 1/8
        s2_0 = self.down_2(s1)
        s2 = self.stage_2(s2_0)
        # 1/16
        s3_0 = self.down_3(s2)
        s3 = self.stage_3(s3_0) 
        # 1/32
        s4_0 = self.down_4(s3)
        s4 = self.stage_4(s4_0) 


        level_global = self.global_pool(s4)
        level_global = self.tail_up(level_global)


        level_32 = self.arm_32(s4)
        level_32 = level_32 + level_global
        level_32 = self.level_32_up(level_32)

        level_16 = self.arm_16(s3)
        level_16 = self.level_16_up(level_16)

        context = level_16+level_32

        feature = torch.cat((spatial, context), 1)
        
        heatmap = self.ffm(feature)
        out = self.up(heatmap)
    
        return out
