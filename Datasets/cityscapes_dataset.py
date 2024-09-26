import os, json
import cv2, glob
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
    Label('unlabeled', 0, -1, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, -1, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, -1, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, -1, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, -1, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, -1, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, -1, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, -1, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, -1, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, -1, 'construction', 2, False, True, (180, 165, 180)),
    Label('bridge', 15, -1, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, -1, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, -1, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, -1, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, -1, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', 34, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]
# name to label object
name2label = {label.name: label for label in labels}
# id to label object
id2label = {label.id: label for label in labels}
# trainId to label object
trainId2label = {label.trainId: label for label in reversed(labels)}



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
        mask = Image.open(os.path.join(self.args.mask_dir, self.split, city_name, img_name[:-16]+'_gtFine_labelIds.png'))
        mask = np.array(mask)
        for i in range(len(id2label)):
            mask[mask == i] = id2label[i].trainId +1
        mask = Image.fromarray(mask.astype(np.uint8))

        sample = [img, mask]

        sample = self.transform(sample)

        return sample, str(img_name)
    def __len__(self):
        return self.num_samples

if __name__ == '__main__':


    print(name2label)
    print(trainId2label)
    print(id2label[5])