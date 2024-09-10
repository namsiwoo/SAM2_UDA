import os, json

import numpy as np
import torch.nn as nn
import torch.utils.data
from utils.get_transforms import get_transforms

from PIL import Image

import matplotlib.pyplot as plt

class synthia_dataset(torch.utils.data.Dataset): #MO, CPM, CoNSeP
    def __init__(self, args, split):
        self.args = args
        self.split = split

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        if self.split == 'train':
            self.transform = get_transforms({
                'random_resize': [0.8, 1.25],
                'horizontal_flip': False,
                'random_affine': 0.3,
                'random_rotation': 30,
                'random_crop': (512, 1024),
                'to_tensor': 2, # number of img
                'normalize': np.array([self.mean, self.std])
            })
        else:
            self.transform = get_transforms({
                'to_tensor': 2,
                'normalize': np.array([self.mean, self.std])
            })

        # read samples
        self.samples = self.read_samples(self.split)

        # set num samples
        self.num_samples = len(self.samples)
        print('{} dataset {} loaded'.format(self.split, self.num_samples))


    def read_samples(self, split):
        with open(os.path.join(self.args.img2_dir, 'train_val_test.json')) as f:
            split_dict = json.load(f)
        filename_list = split_dict[split]

        return filename_list

    def __getitem__(self, index):
        img_name = self.samples[index % len(self.samples[1])]

        img1 = Image.open(os.path.join(self.args.img1_dir, img_name)).convert('RGB')
        img2 = Image.open(os.path.join(self.args.img2_dir, 'vis', img_name)).convert('RGB')
        mask = Image.open(os.path.join(self.args.mask_dir, img_name)).convert('L')

        sample = [img1, img2, mask]

        sample = self.transform(sample)

        return sample, str(img_name)
    def __len__(self):
        return self.num_samples


