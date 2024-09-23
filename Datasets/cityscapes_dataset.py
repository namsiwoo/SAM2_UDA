import os, json
import cv2, glob
import numpy as np
import torch.nn as nn
import torch.utils.data
from utils.get_transforms import get_transforms

from PIL import Image

import matplotlib.pyplot as plt

class cityscapes_dataset(torch.utils.data.Dataset):
    def __init__(self, args, split):
        self.args = args
        self.split = split

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        if self.split == 'train':
            self.transform = get_transforms({
                # 'random_resize': [0.8, 1.25],
                # 'horizontal_flip': False,
                # 'random_affine': 0.3,
                # 'random_rotation': 30,
                'random_crop': (512, 1024),
                'to_tensor': 1, # number of img
                'normalize': np.array([self.mean, self.std])
            })
        else:
            self.transform = get_transforms({
                'to_tensor': 1,
                'normalize': np.array([self.mean, self.std])
            })

        # read samples
        self.samples = glob.glob(os.path.join(self.args.img_dir, split, '*', '*.png'))

        # set num samples
        self.num_samples = len(self.samples)
        print('{} dataset {} loaded'.format(self.split, self.num_samples))

    def __getitem__(self, index):
        img_name = self.samples[index % len(self.samples[1])]

        img = Image.open(img_name).convert('RGB')

        city_name = img_name.split('/')[-2]
        img_name = img_name.split('/')[-1]
        # mask = cv2.imread(os.path.join(self.args.mask_dir, img_name), cv2.IMREAD_UNCHANGED)
        mask = Image.open(os.path.join(self.args.mask_dir, self.split, city_name, img_name[:-4]+'_gtFine_labelIds.png'))

        sample = [img, mask]

        sample = self.transform(sample)

        return sample, str(img_name)
    def __len__(self):
        return self.num_samples


