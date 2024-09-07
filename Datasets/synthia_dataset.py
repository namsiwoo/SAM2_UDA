import os, json

import numpy as np
import torch.utils.data

from PIL import Image

import matplotlib.pyplot as plt

class syenthia_dataset(torch.utils.data.Dataset): #MO, CPM, CoNSeP
    def __init__(self, args, split):
        self.args = args
        self.split = split

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # read samples
        self.samples = self.read_samples(self.split, few_shot=args.fs)

        # set num samples
        if self.split == 'train':
            self.num_samples = len(self.samples[0])
        else:
            self.num_samples = len(self.samples)
        print('{} dataset {} loaded'.format(self.split, self.num_samples))


    def read_samples(self, split):
        with open(os.path.join(self.args.mask_dir, 'train_val_test.json')) as f:
            split_dict = json.load(f)
        filename_list = split_dict[split]
        samples = [os.path.join(f) for f in filename_list]

        return samples

    def __getitem__(self, index):
        if self.split == 'train':
            img_name = self.samples[1][index % len(self.samples[1])]
            img2 = Image.open(os.path.join(self.data2, self.path2, self.split, img_name)).convert('RGB')

            img_name = self.samples[0][index % len(self.samples[0])]
            img1 = Image.open(os.path.join(self.data1, self.path1, self.split, img_name)).convert('RGB')
            if self.use_mask == True:
                box_label = np.array(Image.open(os.path.join(self.data1, 'labels_instance', self.split, img_name[:-4]+self.ext1)))
                box_label = skimage.morphology.label(box_label)
                box_label = Image.fromarray(box_label.astype(np.uint16))
                sample = [img1, img2, box_label]
            else:
                point = Image.open(os.path.join(self.data1, 'labels_point', self.split, img_name[:-4]+self.ext1)).convert('L')
                point = binary_dilation(np.array(point), iterations=2)
                point = Image.fromarray(point)
                sample = [img1, img2, point]
        else:
            img_name = self.samples[index % len(self.samples)]
            img2 = Image.open(os.path.join(self.data2, self.path2, self.split, img_name)).convert('RGB')

            mask = Image.open(os.path.join(self.data2, 'labels_instance', self.split, img_name[:-4]+self.ext2))
            sample = [img2, mask]
        sample = self.transform(sample)

        return sample, str(img_name)
    def __len__(self):
        return self.num_samples


img_path = '/media/NAS/nas_187/datasets/synthia/RAND_CITYSCAPES/RGB'