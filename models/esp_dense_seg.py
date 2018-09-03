
import torch
import torch.nn as nn
import torch.nn.functional as F


class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, dilation):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(growthRate)
    def forward(self, x):

        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

class DenseBlock(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, d, reset_channel=False):
        super(DenseBlock, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):    
            modules.append(make_dense(nChannels_, growthRate, dilation=d[i]))
            nChannels_ += growthRate 
        self.dense_layers = nn.Sequential(*modules)

        self.reset_channel = reset_channel
        if  self.reset_channel:
            self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, stride=1,padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        if self.reset_channel:
            out = self.conv_1x1(out)

        return out
# DilatedDenseResidualBlock
class DDRB(nn.Module):
    def __init__(self, nIn, s=4, d =[1,2,4], add=True):

        super().__init__()

        n = int(nIn//s) 

        self.conv =  nn.Conv2d(nIn, n, 1, stride=1, padding=0, bias=False)
        self.dense_block = DenseBlock(n, nDenselayer=(s-1), growthRate=n, d=d)

        self.bn = nn.BatchNorm2d(nIn)
        self.act = nn.ReLU()

        self.add = add

    def forward(self, input):

        # reduce
        inter = self.conv(input)
        dense_out =self.dense_block(inter)

        # if residual version
        if self.add:
            combine = input + dense_out
        output = self.act(self.bn(combine))
        return output

# DilatedDenseBlock
class DDB(nn.Module):

    def __init__(self, nIn, d =[1,2,4]):

        super().__init__()

        self.dense_block = DenseBlock(nIn, nDenselayer=3, growthRate=nIn, d=d, reset_channel=True)

        self.bn = nn.BatchNorm2d(nIn)
        self.act = nn.ReLU()


    def forward(self, input):

        # reduce
        dense_out =self.dense_block(input)
        output = self.act(self.bn(dense_out))
        return output

# DilatedParllelResidualBlock
class DPRB(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''
    def __init__(self, nIn, add=True):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()

        # s = 2 dowm
        n = int(nIn//6) 

        self.conv =  nn.Conv2d(nIn, n, 3, stride=1, padding=1, bias=False)

        self.d0  = nn.Conv2d(n, n, 1, 1, 0, bias=False) # conv 1x1 
        self.d1  = nn.Conv2d(n, n, 3, 1, padding=1,  dilation=1,  bias=False) # dilation rate of 2^0
        self.d2  = nn.Conv2d(n, n, 3, 1, padding=2,  dilation=2,  bias=False) # dilation rate of 2^1
        self.d4  = nn.Conv2d(n, n, 3, 1, padding=4,  dilation=4,  bias=False) # dilation rate of 2^2
        self.d8  = nn.Conv2d(n, n, 3, 1, padding=8,  dilation=8,  bias=False) # dilation rate of 2^3
        self.d16 = nn.Conv2d(n, n, 3, 1, padding=16, dilation=16, bias=False) # dilation rate of 2^4

        self.bn = nn.BatchNorm2d(nIn, eps=1e-03)
        self.act = nn.PReLU(nIn)

        self.add = add

    def forward(self, input):

        # reduce
        inter = self.conv(input)

        d0 = self.d0(inter)
        # split and transform
        d1 = self.d1(inter)
        d2 = self.d2(inter)
        d4 = self.d4(inter)
        d8 = self.d8(inter)
        d16 = self.d16(inter)

        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
   
        #merge
        combine = torch.cat([d0, d1, add1, add2, add3, add4], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.act(self.bn(combine))
        return output


class ESPD_SegNet(nn.Module):

    def __init__(self, classes=1, p=1, q=1, r=2, t=2):
        '''
        :param classes: number of classes in the dataset.
        '''
        super().__init__()

        # -----------------------------------------------------------------
        # encoder 
        # ---------------------
        self.conv0 = nn.Conv2d(3, 12, 3, stride=1, padding=1, bias=False)

        self.b0 = nn.BatchNorm2d(12, eps=1e-03)
        self.a0 = nn.PReLU(12)

        self.down_1 = nn.Conv2d(12, 12, 3, stride=2, padding=1, bias=False)
        self.stage_1_0 = DPRB(12, add=True)
        block = [DDRB(12, s=6, d=[1,2,4,6,8], add=True) for _ in range(p)]
        self.stage_1 = nn.Sequential(*block)

        self.b1 = nn.BatchNorm2d(24, eps=1e-03)
        self.a1 = nn.PReLU(24)

        self.down_2 = nn.Conv2d(24, 24, 3, stride=2, padding=1, bias=False)
        self.stage_2_0 = DPRB(24, add=True)
        block = [DDRB(24, s=6, d=[1,2,4,6,8], add=True) for _ in range(q)]
        self.stage_2 = nn.Sequential(*block)

        self.b2 = nn.BatchNorm2d(48, eps=1e-03)
        self.a2 = nn.PReLU(48)

        self.down_3 = nn.Conv2d(48, 48, 3, stride=2, padding=1, bias=False)
        self.stage_3_0 = DPRB(48, add=True)
        block = [DDRB(48, s=6, d=[1,2,4,6,8], add=True) for _ in range(r)]
        self.stage_3 = nn.Sequential(*block)

        self.b3 = nn.BatchNorm2d(96, eps=1e-03)
        self.a3 = nn.PReLU(96)

        self.down_4 = nn.Conv2d(96, 96, 3, stride=2, padding=1, bias=False)
        self.stage_4_0 = DPRB(96, add=True)
        block = [DDRB(96, s=6, d=[1,2,4,6,8], add=True) for _ in range(t)]
        self.stage_4 = nn.Sequential(*block)

        self.b4 = nn.BatchNorm2d(192, eps=1e-03)
        self.a4 = nn.PReLU(192)

        # -----------------------------------------------------------------
        # heatmap 
        # ---------------------

        self.classifier = nn.Conv2d(192, 1, 1, stride=1, padding=0, bias=False)

        # -----------------------------------------------------------------
        # decoder 
        # ---------------------
        self.bn_ = nn.BatchNorm2d(1, eps=1e-03)
        self.relu_ = nn.ReLU()

        self.stage3_down = nn.Conv2d(96, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.stage2_down = nn.Conv2d(48, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.stage1_down = nn.Conv2d(24, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_3 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
 
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

        x = self.conv0(input)
        x = self.a0(self.b0(x))
        # ---------------
        # pool
        s1_pool = self.down_1(x)
        s1_0 = self.stage_1_0(s1_pool)
        s1 = self.stage_1(s1_0)
        # concat
        concat_1 = torch.cat((s1, s1_0), dim=1)
        concat_1 = self.a1(self.b1(concat_1))
        # ---------------
        # pool
        s2_pool = self.down_2(concat_1)
        s2_0 = self.stage_2_0(s2_pool)
        s2 = self.stage_2(s2_0)
        # concat
        concat_2 = torch.cat((s2, s2_0), dim=1)
        concat_2 = self.a2(self.b2(concat_2))
        # ---------------
        # pool
        s3_pool = self.down_3(concat_2)
        s3_0 = self.stage_3_0(s3_pool)
        s3 = self.stage_3(s3_0)
        # concat
        concat_3 = torch.cat((s3, s3_0), dim=1)
        concat_3 = self.a3(self.b3(concat_3))
        # ---------------
        # pool
        s4_pool = self.down_4(concat_3)
        s4_0 = self.stage_4_0(s4_pool)
        s4 = self.stage_4(s4_0)
        # concat
        concat_4 = torch.cat((s4, s4_0), dim=1)
        concat_4 = self.a4(self.b4(concat_4))


        heatmap = self.classifier(concat_4)

        
        # heatmap_1 = self.bn_(self.up1(heatmap))
        heatmap_1 = F.upsample(heatmap, scale_factor=2, mode='bilinear', align_corners=True)  
        s3_heatmap = self.bn_(self.stage3_down(concat_3))
        heatmap_1 = heatmap_1 + s3_heatmap
        heatmap_1 = self.conv_1(heatmap_1)

        # heatmap_2 = self.bn_(self.up2(heatmap_1))
        heatmap_2 = F.upsample(heatmap_1, scale_factor=2, mode='bilinear', align_corners=True)  
        s2_heatmap = self.bn_(self.stage2_down(concat_2))
        heatmap_2 = heatmap_2 + s2_heatmap
        heatmap_2 = self.conv_2(heatmap_2)

        # heatmap_3 = self.bn_(self.up3(heatmap_2))
        heatmap_3 = F.upsample(heatmap_2, scale_factor=2, mode='bilinear', align_corners=True)  
        s1_heatmap = self.bn_(self.stage1_down(concat_1))
        heatmap_3 = heatmap_3 + s1_heatmap
        heatmap_3 = self.conv_3(heatmap_3)

        out = F.upsample(heatmap_3, scale_factor=2, mode='bilinear', align_corners=True)  
        # out = self.up4(heatmap_3)
        # return heatmap, heatmap_1, heatmap_2, heatmap_3, out
        return out




