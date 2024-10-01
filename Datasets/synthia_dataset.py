import os, json
import cv2
import numpy as np
import torch.nn as nn
import torch.utils.data
from utils.get_transforms import get_transforms

from PIL import Image

import matplotlib.pyplot as plt

from collections import namedtuple
Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',  # Whether this label distinguishes between single instances or not

    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])

# --------------------------------------------------------------------------------
# A list of all labels
# --------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label('void', 0, 0, 'void', 0, False, True, (0, 0, 0)),
    Label('sky', 1, 11, 'void', 0, False, True, (0, 0, 0)),
    Label('building', 2, 3, 'void', 0, False, True, (0, 0, 0)),
    Label('road', 3, 1, 'void', 0, False, True, (0, 0, 0)),
    Label('sidewalk', 4, 2, 'void', 0, False, True, (0, 0, 0)),
    Label('fence', 5, 5, 'void', 0, False, True, (0, 0, 0)),
    Label('vegetation', 6, 9, 'void', 0, False, True, (111, 74, 0)),
    Label('pole', 7, 6, 'void', 0, False, True, (81, 0, 81)),
    Label('car', 8, 14, 'flat', 1, False, False, (128, 64, 128)),
    Label('traffic sign', 9, 8, 'flat', 1, False, False, (244, 35, 232)),
    Label('pedestrian', 10, 0, 'flat', 1, False, True, (250, 170, 160)),
    Label('bicycle', 11, 19, 'flat', 1, False, True, (230, 150, 140)),
    Label('motorcycle', 12, 18, 'construction', 2, False, False, (70, 70, 70)),
    Label('parking-slot', 13, 0, 'construction', 2, False, False, (102, 102, 156)),
    Label('road-work', 14, 12, 'construction', 2, False, False, (190, 153, 153)),
    Label('traffic light', 15, 7, 'construction', 2, False, True, (180, 165, 180)),
    Label('terrain', 16, 10, 'construction', 2, False, True, (150, 100, 100)),
    Label('rider', 17, 13, 'construction', 2, False, True, (150, 120, 90)),
    Label('truck', 18, 15, 'object', 3, False, False, (153, 153, 153)),
    Label('bus', 19, 16, 'object', 3, False, True, (153, 153, 153)),
    Label('train', 20, 17, 'object', 3, False, False, (250, 170, 30)),
    Label('wall', 21, 4, 'object', 3, False, False, (220, 220, 0)),
    Label('lanemarking', 22, 0, 'nature', 4, False, False, (107, 142, 35)),
]
# name to label object
name2label = {label.name: label for label in labels}
# id to label object
id2label = {label.id: label for label in labels}
# trainId to label object
trainId2label = {label.trainId: label for label in reversed(labels)}

class synthia_dataset(torch.utils.data.Dataset): #NO Sky
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
                'to_tensor': 2, # number of img
                # 'normalize': np.array([self.mean, self.std])
            })
        else:
            self.transform = get_transforms({
                'to_tensor': 2,
                # 'normalize': np.array([self.mean, self.std])
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
        # img2 = Image.open(os.path.join(self.args.img2_dir, 'vis_default', img_name)).convert('RGB')

        mask = cv2.imread(os.path.join(self.args.mask_dir, img_name), cv2.IMREAD_UNCHANGED)
        # mask = Image.open(os.path.join(self.args.mask_dir, img_name)).convert('I;16')
        mask = mask[:, :, 2].astype(np.uint8)

        for i in range(len(id2label)):
            mask[mask == i] = id2label[i].trainId
        mask = Image.fromarray(mask.astype(np.uint8))


        sample = [img1, img2, mask]

        sample = self.transform(sample)
        # print(torch.min(sample[0]), torch.min(sample[1]), torch.max(sample[0]), torch.max(sample[1]))

        return sample, str(img_name)
    def __len__(self):
        return self.num_samples


