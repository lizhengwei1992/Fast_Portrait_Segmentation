# Fast_Portrait_Segmentation
Fast (aimed to "real time") Portrait Segmentation at mobile phone

This project is not normal semantic segmentation but focus on **real-time protrait segmentation**.All the experimentals works with **pytorch**.


I hope to find a effcient network which can run on **mobile phone**. Currently, successfull application of person body/protrait segmentation can be find in APP like **SNOW**&**B612**, whose technology is proposed by a Korea company [Nalbi](https://www.nalbi.ai/).


# Models

- ## [mobilenet_dilate_unet](https://github.com/lizhengwei1992/Fast_Portrait_Segmentation/blob/master/models/mv2_dilate_unet.py)<sup>[1][2][7][9]</sup>

    Encoder : mobilenet_v2(os: 32) 
    
    Decoder : unet(concat low level feature)
             use dilate convolution at different stage(d = 2, 6, 12, 18)
             
- ## [Shuffle_Seg_SkipNet](https://github.com/lizhengwei1992/Fast_Portrait_Segmentation/blob/master/models/shuffle_seg_skipnet.py)<sup>[4][10][18]</sup>

    Encoder : shufflenet
    
    Decoder : skip connection (add low level feature)
    
- ## [esp_dense_seg](https://github.com/lizhengwei1992/Fast_Portrait_Segmentation/blob/master/models/esp_dense_seg.py)<sup>[20][10][15][19]</sup>


- ## [residualdense_bisenet](https://github.com/lizhengwei1992/Fast_Portrait_Segmentation/blob/master/models/residualdense_bisenet.py)<sup>[15][23][24]</sup>

    Attention model is a potential module in the segmentation task. I use a very light residual-dense net as the backbone of the Context Path. The details about fussion of last features in Contxt Path is not clear in the paper(BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation). 

- ## Segmentation + Matting <sup>[7][12][15]</sup>
    Hard segmentation + Soft matting.(coming soon)





# Speed Analysis
:zap: **Real-time ! ! !** :tada::tada::tada:

Platform    : [ncnn](https://github.com/Tencent/ncnn).

Mobile phone: Samsung Galaxy S8+(cpu).


|            | model size (M) | time(ms)      | 
| ---------- | :-----------:  | :-----------: |
| model_seg_matting    |          3.3     |     ~40          |

***update : 2018/12/27***: Demo video on my iphone 6 ([baiduyun](https://pan.baidu.com/s/1nieS7dSMw6Kwzsa1dz4egA))


# Result Examples

HUAWEI Mate 20 released recently can keep color on human and make the bacgrand gray in real time ([click to view](https://www.bilibili.com/video/av34321080/?spm_id_from=333.788.videocard.1) ). I test my model using cpu on my MAC, getting some videos here.

<img src="https://github.com/lizhengwei1992/Fast_Portrait_Segmentation/raw/master/result/1.gif" width="480" height="270" >
<img src="https://github.com/lizhengwei1992/Fast_Portrait_Segmentation/raw/master/result/2.gif" width="480" height="270" >
<img src="https://github.com/lizhengwei1992/Fast_Portrait_Segmentation/raw/master/result/3.gif" width="480" height="270" >
<img src="https://github.com/lizhengwei1992/Fast_Portrait_Segmentation/raw/master/result/4.gif" width="480" height="270" >








# References
## papers
- [1]  [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)
- [2]  [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf)
- [3]  [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/pdf/1707.01083.pdf)
- [4]  [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/pdf/1807.11164.pdf)
- [5]  [CondenseNet: An Efficient DenseNet using Learned Group Convolutions](https://arxiv.org/pdf/1711.09224.pdf)
- [6]  [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf)
- [7]  [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)
- [8]  [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1706.05587.pdf)
- [9]  [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
- [10] [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1411.4038.pdf)
- [11] [Automatic Portrait Segmentation for Image Stylization](http://xiaoyongshen.me/webpage_portrait/papers/portrait_eg16.pdf)
- [12] [Fast Deep Matting for Portrait Animation on Mobile Phone](https://arxiv.org/pdf/1707.08289.pdf)
- [13] [DenseASPP for Semantic Segmentation in Street Scenes](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf)
- [14] [Learning a Discriminative Feature Network for Semantic Segmentation](https://arxiv.org/pdf/1804.09337.pdf)
- [15] [ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation](http://www.robesafe.es/personal/eduardo.romera/pdfs/Romera17tits.pdf)
- [16] [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/pdf/1606.02147.pdf)
- [17] [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/pdf/1511.00561.pdf)
- [18] [ICNet for Real-Time Semantic Segmentation on High-Resolution Image](https://arxiv.org/pdf/1704.08545.pdf)
- [19] [ShuffleSeg: Real-time Semantic Segmentation Network](https://arxiv.org/pdf/1803.03816.pdf)
- [20] [ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation](https://arxiv.org/pdf/1803.06815.pdf)
- [21] [Efficient Semantic Segmentation using Gradual Grouping](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w12/Vallurupalli_Efficient_Semantic_Segmentation_CVPR_2018_paper.pdf)
- [22] [Analysis of efficient CNN design techniques for semantic segmentation](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w12/Briot_Analysis_of_Efficient_CVPR_2018_paper.pdf)
- [23] [Dual Attention Network for Scene Segmentation](https://arxiv.org/pdf/1809.02983.pdf)
- [24] [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/pdf/1808.00897.pdf)
- [25] [CCNet: Criss-Cross Attention for Semantic Segmentation](https://arxiv.org/pdf/1811.11721.pdf)
