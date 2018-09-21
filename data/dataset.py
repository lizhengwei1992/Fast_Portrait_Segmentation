'''
dataset: 

Author: Zhengwei Li
Data: July 20 2018
'''

from __future__ import print_function, division
import os

import numpy as np
from data import data_config
import torch
import cv2

INPUT_SIZE = 512


class coco(data.Dataset):

    def __init__(self,
                 base_dir='',
                 ):
        """
			dataset: portrait segmentation
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'images')
        self._label_dir = os.path.join(self._base_dir, 'masks')

        self.img_List = os.listdir(self._image_dir)
        self.lal_List = os.listdir(self._label_dir)
        self.img_List.sort()
        self.lal_List.sort()

        self.data_num = len(self.img_List)

        print("Dataset : ulsee_coco !")
        print('file number %d' % len(self.img_List))


    def __getitem__(self, index):

        _img = cv2.imread(os.path.join(self._image_dir, self.img_List[index])).astype(np.float32)
        _img = (_img - (104., 112., 121.,)) / 255.0
        _target = cv2.imread(os.path.join(self._label_dir, self.lal_List[index])).astype(np.float32) #(0,1)

        _img, _target = data_config.crop_patch(_img, _target, INPUT_SIZE)

        _img, _target = data_config.augment(_img, _target)

        _img, _target = data_config.np2Tensor(_img, _target)
        _target = _target[0,:,:].unsqueeze_(0)

        sample = {'image': _img, 'gt': _target}

        return sample

    def __len__(self):
        return self.data_num
        
        
 
