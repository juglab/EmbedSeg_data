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


class CVPPP2014Dataset(Dataset):

    class_ids= (0)
    def __init__(self, root_dir='./', type="train", bg_id=0, size=None, transform=None): #TODO: 26

        print('CVPPP2014 Dataset Reduced Dataset created')

        # get image and instance list
        image_list=glob.glob(os.path.join(root_dir, 'cvppp2014/final-0/{}/'.format(type), 'images/*.tif'))
        image_list.sort()
        self.image_list = image_list

        instance_list=glob.glob(os.path.join(root_dir, 'cvppp2014/final-0/{}/'.format(type), 'masks/*.tif'))
        instance_list.sort()
        self.instance_list = instance_list

        self.bg_id = bg_id
        self.size = size
        self.real_size = len(self.image_list)
        self.transform = transform

    def __len__(self):

        return self.real_size if self.size is None else self.size

    def __getitem__(self, index):

        index = index if self.size is None else random.randint(0, self.real_size-1)
        sample = {}

        # load image
        image=tifffile.imread(self.image_list[index])
        #sample['image'] = image
        sample['image']= torch.from_numpy(np.transpose(image, (2, 0, 1))).float().div(255)
        #sample['image'] = torch.from_numpy(image[np.newaxis, ...]).float().div(255)
        sample['im_name'] = self.image_list[index]

        # load instances
        instance = tifffile.imread(self.instance_list[index])
        instance, label = self.decode_instance(instance, self.bg_id)

        #sample['instance'] = instance
        #sample['label'] = label

        sample['instance'] = torch.from_numpy(instance[np.newaxis, ...]).byte()  # TODO
        sample['label'] = torch.from_numpy(label[np.newaxis, ...]).byte()  # TODO

        return sample

    @classmethod
    def decode_instance(cls, pic, bg_id=None):
        pic = np.array(pic, copy=False)

        instance_map = np.zeros(
            (pic.shape[0], pic.shape[1]), dtype=np.uint8)

        # contains the class of each instance, but will set the class of "unlabeled instances/groups" to bg
        class_map = np.zeros(
            (pic.shape[0], pic.shape[1]), dtype=np.uint8)

        if bg_id is not None:
            mask = pic > bg_id

            if mask.sum() > 0:
                ids, _, _ = relabel_sequential(pic[mask])

                instance_map[mask] = ids
                class_map[mask] = 1

        return instance_map, class_map
