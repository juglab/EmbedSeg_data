"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import glob
import os
import random

import numpy as np
import pandas as pd
from PIL import Image
from skimage.segmentation import relabel_sequential
import tifffile
import torch
from torch.utils.data import Dataset


class Usiigaci2019Dataset(Dataset):

    def __init__(self, root_dir='./', type="train", bg_id=0, centre='com', size=None, transform=None):

        print('Usiigaci 2019 Dataset created')

        image_list = glob.glob(os.path.join(root_dir, 'usiigaci512x5122019/final-0/{}/'.format(type), 'images/*.tif'))  # TODO
        image_list.sort()
        self.image_list = image_list

        instance_list = glob.glob(os.path.join(root_dir, 'usiigaci512x5122019/final-0/{}/'.format(type), 'masks/*.tif'))  # TODO

        instance_list.sort()
        self.instance_list = instance_list

        self.bg_id = bg_id
        self.size = size
        self.real_size = len(self.image_list)
        self.transform = transform
        # xm = np.linspace(0, 2, 2048).reshape(
        #     1, 1, -1).repeat(1, 1024, 2048)
        # ym = np.linspace(0, 1, 1024).reshape(
        #     1, -1, 1).repeat(1, 1024, 2048)
        xm = np.repeat(np.linspace(0, 2, 2048).reshape(1, 1, -1), 1024, 1)
        ym = np.repeat(np.linspace(0, 1, 1024).reshape(1, -1, 1), 2048, 2)
        xym=np.concatenate((xm, ym), 0)
        self.xym_s = np.ascontiguousarray(xym[:, :512, :512])  # is contiguous needed?
        self.type=type
        self.centre=centre

    def __len__(self):

        return self.real_size if self.size is None else self.size


    def __getitem__(self, index):

        index = index if self.size is None else random.randint(0, self.real_size-1)
        sample = {}

        # load image
        image=tifffile.imread(self.image_list[index]).astype("float32")
        sample['image'] = torch.from_numpy(image[np.newaxis, ...]).float().div(65535)
        sample['im_name'] = self.image_list[index]

        # load instances
        instance = tifffile.imread(self.instance_list[index])
        #instance, label, centre_map = self.decode_instance(instance, self.xym_s, self.bg_id, self.centre)
        instance, label, ids, centre_x, centre_y = self.decode_instance(instance, self.xym_s, self.bg_id, self.centre)
        sample['instance'] = torch.from_numpy(instance[np.newaxis, ...]).byte()  # TODO
        sample['label'] = torch.from_numpy(label[np.newaxis, ...]).byte()  # TODO
        #sample['centre_map']=torch.from_numpy(centre_map[np.newaxis, ...]).float()  # TODO
        sample['ids'] =torch.from_numpy(ids.astype(np.int16)).byte()
        sample['centre_x']=torch.from_numpy(centre_x).float()
        sample['centre_y']=torch.from_numpy(centre_y).float()
        return sample

    @classmethod
    def decode_instance(cls, pic, xym_s, bg_id=None, centreType='com'):
        pic = np.array(pic, copy=False)

        instance_map = np.zeros(
            (pic.shape[0], pic.shape[1]), dtype=np.uint8)
        #centre_map=np.zeros((2, pic.shape[0], pic.shape[1]), dtype=np.float32)
        # contains the class of each instance, but will set the class of "unlabeled instances/groups" to bg
        class_map = np.zeros(
            (pic.shape[0], pic.shape[1]), dtype=np.uint8)

        if bg_id is not None:
            mask = pic > bg_id

            if mask.sum() > 0:
                ids, _, _ = relabel_sequential(pic[mask])

                instance_map[mask] = ids
                class_map[mask] = 1


        centre_x=np.zeros((len(ids)))
        centre_y=np.zeros((len(ids)))
        for index, id in enumerate(ids):
            in_mask = instance_map[np.newaxis, ...] == id  # 1 x h x w
            if centreType == 'com':
                #xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)
                xy_in =xym_s[np.repeat(in_mask, 2, axis=0)].reshape(2, -1)
                centre = xy_in.mean(1)
                centre_x[index] = centre[0]
                centre_y[index]= centre[1]
                #centre = xy_in.mean(1).reshape(2, 1, 1)  # 2 x 1 x 1
                #np.putmask(centre_map, np.repeat(in_mask, 2, axis=0), centre.astype(np.float32))
                #center_map[np.repeat(in_mask, 2, axis=0)]=center

        print("Instance done!")
        #return instance_map, class_map, centre_map
        return instance_map, class_map, ids, centre_x, centre_y
