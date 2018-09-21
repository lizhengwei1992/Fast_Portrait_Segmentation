'''
dataconfig: 
Author: Zhengwei Li
Data: July 22 2018
'''

import torch, cv2
import random as r
import numpy as np
import math


## ---------------------------------
def crop_patch(Img, Tar, patch):
    (h, w, c) = Img.shape

    if r.random() < 0.5:
        if h>patch and w>patch:
            x = r.randrange(0, (w - patch))
            y = r.randrange(0, (h - patch))

            Img = Img[y:y + patch, x:x + patch, :]
            Tar = Tar[y:y + patch, x:x + patch, :]
        else:
            Img = cv2.resize(Img, (patch,patch), interpolation=cv2.INTER_CUBIC)
            Tar = cv2.resize(Tar, (patch,patch), interpolation=cv2.INTER_NEAREST)
    else:
        Img = cv2.resize(Img, (patch,patch), interpolation=cv2.INTER_CUBIC)
        Tar = cv2.resize(Tar, (patch,patch), interpolation=cv2.INTER_NEAREST) 
              
    return Img, Tar


def augment(Img, Tar):
    if r.random() < 0.5:
        Img = Img[:, ::-1, :]
        Tar = Tar[:, ::-1, :]
    if r.random() < 0.5:
        Img = Img[::-1, :, :]
        Tar = Tar[::-1, :, :]
    return Img, Tar

def np2Tensor(Img, Tar):
    ts = (2, 0, 1)

    Img = torch.FloatTensor(Img.transpose(ts).astype(float)).mul_(1.0)
    Tar = torch.FloatTensor(Tar.transpose(ts).astype(float)).mul_(1.0)
    
    return Img, Tar

