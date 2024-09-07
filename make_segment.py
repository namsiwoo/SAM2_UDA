import os, random, json
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image

import torch

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

np.random.seed(3)

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
    mask = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))

    for i, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3)])
        img[m] = color_mask
        mask[m] = i+1

    return img, mask

def make_split(data_root, save_root):
    img_list = os.listdir(os.path.join(data_root, 'RGB'))
    random.shuffle(img_list)

    train_list = img_list[:int(len(img_list)*0.5)]
    val_list = img_list[int(len(img_list)*0.5): int(len(img_list)*0.7)]
    test_list = img_list[int(len(img_list)*0.7):]

    json_data = OrderedDict()
    json_data['train'] = train_list
    json_data['val'] = val_list
    json_data['test'] = test_list

    with open('{:s}/train_val_test.json'.format(save_root), 'w') as make_file:
        json.dump(json_data, make_file, indent='\t')
def main(data_root, save_root):
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    sam2_checkpoint = "/media/NAS/nas_187/siwoo/2024/UDA_citycapes/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)


    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    # mask_generator_2 = SAM2AutomaticMaskGenerator(
    #     model=sam2,
    #     points_per_side=64,
    #     points_per_batch=128,
    #     pred_iou_thresh=0.7,
    #     stability_score_thresh=0.92,
    #     stability_score_offset=0.7,
    #     crop_n_layers=1,
    #     box_nms_thresh=0.7,
    #     crop_n_points_downscale_factor=2,
    #     min_mask_region_area=25.0,
    #     use_m2m=True,
    # )

    img_list = os.listdir(os.path.join(data_root, 'RGB'))
    for img_name in img_list:
        img_path = os.path.join(data_root, img_name)
        img = np.array(Image.open(img_path).convert('RGB'))
        mask = mask_generator.generate(img)
        img, mask = show_anns(mask)

        img = Image.fromarray((img*255).astype(np.uint8))
        img.save(os.path.join(save_root, 'vis', img_name))

        mask = Image.fromarray((mask).astype(np.uint16))
        mask.save(os.path.join(save_root, 'masks', img_name))


if __name__ == '__main__':
    data_root = '/media/NAS/nas_187/datasets/synthia/RAND_CITYSCAPES'
    save_root = '/media/NAS/nas_187/siwoo/2024/UDA_citycapes/synthia'
    make_split(data_root, save_root)
    main(data_root, save_root)

    #CUDA_VISIBLE_DEVICES=0 python make_segment.py