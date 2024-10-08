import torch
import os, random, gc, yaml
import argparse
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from Datasets.cityscapes_dataset import cityscapes_dataset
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from utils.utils import colorEncode, save_checkpoint
from utils.compute_iou import SoftDiceLoss, fast_hist, per_class_iu

import csv
from scipy.io import loadmat

colors = loadmat('/media/NAS/nas_70/siwoo_data/UDA_citycapes/color150.mat')['colors']
# names = {}
# with open('data/object150_info.csv') as f:
#     reader = csv.reader(f)
#     next(reader)
#     for row in reader:
#         names[int(row[0])] = row[5].split(";")[0]

def split_forward(predictor, sam2_model, input, num_classes, h_size=512, w_size=1024, device=None):
    # size = 224
    overlap = 80
    _bb_feat_sizes = [
        (256, 256),
        (128, 128),
        (64, 64),
    ]

    b, c, h0, w0 = input.size()

    # zero pad for border patches
    pad_h = 0
    # here the padding is to make the the image size could be divided exactly by the size - overlap (the length of step)
    if h0 - h_size > 0 and (h0 - h_size) % (h_size - overlap) > 0:
        pad_h = (h_size - overlap) - (h0 - h_size) % (h_size - overlap)  # size is the the input size of model
        tmp = torch.zeros((b, c, pad_h, w0))
        input = torch.cat((input, tmp), dim=2)

    if w0 - w_size > 0 and (w0 - w_size) % (w_size - overlap) > 0:  # same as the above
        pad_w = (w_size - overlap) - (w0 - w_size) % (w_size - overlap)
        tmp = torch.zeros((b, c, h0 + pad_h, pad_w))
        input = torch.cat((input, tmp), dim=3)

    _, c, h, w = input.size()

    output = torch.zeros((input.size(0), num_classes, h, w))

    for i in range(0, h - overlap, h_size - overlap):
        r_end = i + h_size if i + h_size < h else h
        ind1_s = i + overlap // 2 if i > 0 else 0
        ind1_e = i + h_size - overlap // 2 if i + h_size < h else h

        for j in range(0, w - overlap, w_size - overlap):
            c_end = j + w_size if j + w_size < w else w

            ind2_s = j + overlap // 2 if j > 0 else 0
            ind2_e = j + w_size - overlap // 2 if j + w_size < w else w

            input_patch = input[:, :, i:r_end, j:c_end]

            with torch.no_grad():
                img = list(input_patch.permute(0, 2, 3, 1).numpy())
                predictor.set_image_batch(img)  # apply SAM image encoder to the image

                sparse_embeddings = torch.empty((len(input_patch), 0, sam2_model.sam_prompt_embed_dim), device=device)
                dense_embeddings = sam2_model.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                    len(input_patch), -1, sam2_model.sam_image_embedding_size, sam2_model.sam_image_embedding_size
                )

                # batched_mode = unnorm_coords.shape[0] > 1  # multi mask prediction
                batched_mode = False  # multi mask prediction

                low_res_masks, prd_scores, _, _ = sam2_model.sam_mask_decoder_ssm(
                    image_embeddings=predictor._features["image_embed"],
                    image_pe=sam2_model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True, repeat_image=batched_mode, high_res_features=predictor._features["high_res_feats"], )

                prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])  # Upscale the masks to the original image resolution

            output[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = prd_masks[:, :, ind1_s - i:ind1_e - i,
                                                     ind2_s - j:ind2_e - j]

    output = output[:, :, :h0, :w0]
    return output

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss

def main(args, device, class_list):
    f = open(os.path.join(args.result, 'log.txt'), 'w')
    f.write('=' * 40)
    f.write('Arguments')
    f.write(str(args))
    f.write('=' * 40)


    import torch

    sam2_checkpoint = "/media/NAS/nas_70/siwoo_data/UDA_citycapes/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l_ssm.yaml"
    # model_cfg = "sam2_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")  # load model
    predictor = SAM2ImagePredictor(sam2_model)
    for name, para in sam2_model.named_parameters():
        if "sam_mask_decoder_ssm" in name:
            para.requires_grad_(True)
        else:
            para.requires_grad_(False)
            # para.requires_grad_(True)

    sam2_model.sam_prompt_encoder.no_mask_embed.requires_grad = True

    for name, p in sam2_model.named_parameters():
        if p.requires_grad:
            print('========', name)

    dropout_modules = [module for module in sam2_model.modules() if isinstance(module, torch.nn.Dropout)]
    [module.eval() for module in dropout_modules]
    # bn_modules = [module for module in net.module.modules() if isinstance(module, torch.nn.BatchNorm2d)]


    optimizer = torch.optim.AdamW(params=sam2_model.parameters(), lr=args.lr) #, weight_decay=4e-5
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=1.0e-7)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    scaler = torch.cuda.amp.GradScaler()  # set mixed precision

    criterion_dice = SoftDiceLoss()
    criterion_ce = torch.nn.CrossEntropyLoss()

    train_dataset = cityscapes_dataset(args, 'train')
    val_dataset = cityscapes_dataset(args, 'val')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, num_workers=8)

    _bb_feat_sizes = [
        (256, 256),
        (128, 128),
        (64, 64),
    ]

    max_miou = 0
    total_train_loss = []
    for epoch in range(args.epochs):
        train_loss = 0
        # predictor.model.train()
        sam2_model.train()
        for iter, batch in enumerate(train_dataloader): # batch[0]
            # with torch.cuda.amp.autocast():  # cast to mix precision
            img = list(batch[0][0].permute(0, 2, 3, 1).detach().numpy())
            mask = batch[0][1].squeeze(1).to(device)

            predictor.set_image_batch(img)  # apply SAM image encoder to the image

            sparse_embeddings = torch.empty((len(img), 0, sam2_model.sam_prompt_embed_dim), device=device)
            dense_embeddings = sam2_model.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                len(img), -1, sam2_model.sam_image_embedding_size, sam2_model.sam_image_embedding_size
            )

            # batched_mode = unnorm_coords.shape[0] > 1  # multi mask prediction
            batched_mode = False  # multi mask prediction
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in
                                 predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = sam2_model.sam_mask_decoder_ssm(
                image_embeddings=predictor._features["image_embed"],
                image_pe=sam2_model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
                multimask_output=True, repeat_image=batched_mode, high_res_features=predictor._features["high_res_feats"], )

            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])  # Upscale the masks to the original image resolution # (2, 20, 512, 1024)
            # prd_masks = F.interpolate(low_res_masks, (mask.shape[1], mask.shape[2]), mode="bilinear", align_corners=False)


            iou_loss = criterion_dice(prd_masks, mask)
            ce_loss = criterion_ce(prd_masks, mask)
            loss = ce_loss + iou_loss
            # loss = cross_entropy2d(prd_masks, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # predictor.model.zero_grad()  # empty gradient ?????????????????????
            # scaler.scale(loss).backward()  # Backpropogate
            # scaler.step(optimizer)
            # scaler.update()  # Mix precision

            lr_scheduler.step()

            train_loss += loss / len(train_dataloader)

            if (iter + 1) % args.print_fq == 0:
                print('{}/{} epoch, {}/{} batch, train loss: {}, ce: {}, iou: {}'.format(epoch,
                                                                                            args.epochs,
                                                                                            iter + 1,
                                                                                            len(train_dataloader),
                                                                                            loss, ce_loss,
                                                                                            iou_loss,
                                                                                            ))

        total_train_loss.append(train_loss)
        print('{} epoch, mean train loss: {}'.format(epoch, total_train_loss[-1]))

        if epoch >= args.start_val:
            sam2_model.eval()
            os.makedirs(os.path.join(args.result, 'img', str(epoch)), exist_ok=True)
            hist = np.zeros((args.num_classes+1, args.num_classes+1))
            ave_mIOUs = []

            with torch.no_grad():
                for iter, batch in enumerate(val_dataloader):
                    img = batch[0][0]
                    prd_masks = split_forward(predictor, sam2_model, img, args.num_classes+1, h_size=512, w_size=1024, device=device)

                    mask = batch[0][1].squeeze(1)[0]
                    img_name = batch[1][0]
                    pred = torch.argmax(prd_masks, dim=1)
                    pred = pred[0].detach().cpu().numpy()

                    hist += fast_hist(mask.numpy().flatten(), pred.flatten(), args.num_classes+1)

                    mIoUs = per_class_iu(hist)
                    ave_mIOUs.append(mIoUs)

                    pred = colorEncode(pred, colors).astype(np.uint8)
                    pred = Image.fromarray((pred).astype(np.uint8))
                    pred.save(os.path.join(args.result, 'img', str(epoch), str(img_name) + '_pred.png'))

                    mask = colorEncode(mask.detach().cpu().numpy(), colors).astype(np.uint8)
                    mask = Image.fromarray((mask).astype(np.uint8))
                    mask.save(os.path.join(args.result, 'img', str(epoch), str(img_name) + '_mask.png'))

                ave_mIOUs = np.mean(np.array(ave_mIOUs), axis=0)
                f = open(os.path.join(args.result, 'img', str(epoch), "result.txt"), 'w')
                f.write('***test result_mask*** class_name\t{:s}\n'.format('\t'.join(class_list)))
                f.write('***test result_mask*** mIOU\t{:s}\n'.format('\t'.join(map(str, ave_mIOUs.tolist()))))
                f.write('***test result_mask*** ave mIOU\t{:s}\n'.format(str(np.nanmean(ave_mIOUs))))
                f.close()

                if max_miou < np.nanmean(ave_mIOUs):
                    print('save {} epoch!!--ave mIOU: {}'.format(str(epoch), np.nanmean(ave_mIOUs)))
                    save_checkpoint(os.path.join(args.result, 'model', 'mIOU_best_model.torch'), predictor.model, epoch)
                    max_miou = np.nanmean(ave_mIOUs)

                print(epoch, ': ave mIOU\t{:s} (b mIOU: {}'.format(str(np.nanmean(ave_mIOUs)), max_miou))

            if max_miou < 15:
                args.start_val = epoch+10


def test(args, device):
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    import torch
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)

    input_channel = 3
    if args.use_sam == True:
        input_channel = 6
    model.backbone.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.classifier[-1] = nn.Conv2d(256, args.num_classes + 1, kernel_size=(1, 1), stride=(1, 1))
    # model.aux_classifier[-1] = nn.Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
    model = model.to(device)

    test_dataseet = synthia_dataset(args, 'train')

    test_dataloader = DataLoader(test_dataseet)
    if args.test_name == "None":
        args.test_name == 'test'
    else:
        args.test_name = 'test_'+args.test_name

    os.makedirs(os.path.join(args.result, 'img',args.test_name), exist_ok=True)
    model.eval()

    hist = np.zeros((args.num_classes + 1, args.num_classes + 1))
    ave_mIOUs = []

    with torch.no_grad():
        for iter, batch in enumerate(test_dataloader):
            img = batch[0][0]
            if args.use_sam == True:
                img2 = batch[0][1]
                img = torch.cat((img, img2), dim=1)

            mask = batch[0][2].squeeze(1).to(device)
            img_name = batch[1][0]
            pred = model(img.to(device))['out']
            pred = torch.argmax(pred, dim=1)
            pred = pred[0].detach().cpu().numpy()

            hist += fast_hist(mask.numpy().flatten(), pred.flatten(), args.num_classes + 1)

            mIoUs = per_class_iu(hist)
            ave_mIOUs.append(mIoUs)

            pred = colorEncode(pred, colors).astype(np.uint8)
            pred = Image.fromarray((pred).astype(np.uint8))
            pred.save(os.path.join(args.result, 'img', args.test_name, str(img_name) + '_pred.png'))

            mask = colorEncode(mask.detach().cpu().numpy(), colors).astype(np.uint8)
            mask = Image.fromarray((mask).astype(np.uint8))
            mask.save(os.path.join(args.result, 'img', args.test_name, str(img_name) + '_mask.png'))

    ave_mIOUs = np.mean(np.array(ave_mIOUs), axis=0)
    f = open(os.path.join(args.result, 'img', args.test_name, "result.txt"), 'w')
    f.write('***test result_mask*** class_name\t{:s}\n'.format('\t'.join(class_list)))
    f.write('***test result_mask*** mIOU\t{:s}\n'.format('\t'.join(map(str, ave_mIOUs.tolist()))))
    f.write('***test result_mask*** ave mIOU\t{:s}\n'.format(str(np.nanmean(ave_mIOUs))))
    f.close()

    print(epoch, ': ave mIOU\t{:s} (b mIOU: {}'.format(str(np.nanmean(ave_mIOUs)), max_miou))

    print('test result: Average- Dice\tIOU\tAJI: '
                 '\t\t{:.4f}\t{:.4f}\t{:.4f}'.format(mean_dice, mean_iou, mean_aji))
    print('test result: Average- DQ\tSQ\tPQ: '
                 '\t\t{:.4f}\t{:.4f}\t{:.4f}'.format(mean_dq, mean_sq, mean_pq))
    # print('test result: Average- AP1\tAP2\tAP3: '
    #              '\t\t{:.4f}\t{:.4f}\t{:.4f}'.format(mean_ap1, mean_ap2, mean_ap3))


    f = open(os.path.join(args.result,'img', args.test_name, "result.txt"), 'w')
    f.write('***test result_mask*** Average- Dice\tIOU\tAJI: '
            '\t\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(mean_dice, mean_iou, mean_aji))
    f.write('***test result_mask*** Average- DQ\tSQ\tPQ: '
            '\t\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(mean_dq, mean_sq, mean_pq))
    # f.write('***test result_mask*** Average- AP1\tAP2\tAP3: '
    #         '\t\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(mean_ap1, mean_ap2, mean_ap3))
    f.close()

    f = open(os.path.join(args.result, "result"+args.test_name[4:]+".txt"), 'w')
    f.write('***test result_mask*** Average- Dice\tIOU\tAJI: '
            '\t\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(mean_dice, mean_iou, mean_aji))
    f.write('***test result_mask*** Average- DQ\tSQ\tPQ: '
            '\t\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(mean_dq, mean_sq, mean_pq))
    # f.write('***test result_mask*** Average- AP1\tAP2\tAP3: '
    #         '\t\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(mean_ap1, mean_ap2, mean_ap3))
    f.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--resume', default=0, type=int, help='')
    parser.add_argument('--start_val', default=5, type=int)
    parser.add_argument('--num_classes', default=12, type=int)
    parser.add_argument('--plt', action='store_true')
    parser.add_argument('--use_sam', action='store_true')

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    parser.add_argument('--model_type', default='deeplabv3_resnet50', help='')
    # parser.add_argument('--img_dir',default='/media/NAS/nas_187/datasets/Cityscapes-Seq/leftImg8bit_sequence_trainvaltest/leftImg8bit_sequence',help='')
    # parser.add_argument('--mask_dir',default='/media/NAS/nas_187/datasets/synthia/RAND_CITYSCAPES/GT/LABELS')
    parser.add_argument('--img_dir',default='/media/NAS/nas_187/datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit',help='')
    parser.add_argument('--mask_dir',default='/media/NAS/nas_187/datasets/cityscapes/gtFine_trainvaltest/gtFine')


    parser.add_argument('--epochs', default=100, type=int, help='')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--lr', default=1e-4, type=float)

    parser.add_argument('--print_fq', default=700, type=int, help='')

    parser.add_argument('--result', default='/media/NAS/nas_70/siwoo_data/UDA_result/SAM2_gt', help='')
    parser.add_argument('--test_name', default='None', type=str, help='')

    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()


    if args.result != None:
        os.makedirs(os.path.join(args.result, 'img'), exist_ok=True)
        os.makedirs(os.path.join(args.result, 'model'), exist_ok=True)

    if args.model_type == 'deeplabv3_resnet50':
        args.model_path = '/media/NAS/nas_70/siwoo_data/UDA_citycapes/best_deeplabv3_resnet50_voc_os16.pth'

    # synthia
    # class_list = ['void', 'sky', 'building', 'road', 'sidewalk', 'fence', 'vegetation', 'pole', 'car', 'traffic sign',
    #               'pedestrian', 'bicycle', 'motorcycle', 'parking-slot', 'road-work', 'traffic light', 'terrain',
    #               'rider', 'truck', 'bus', 'train', 'wall', 'lanemarking']

    class_list = ['void', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
                  'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

    print('=' * 40)
    print(' ' * 14 + 'Arguments')
    for arg in sorted(vars(args)):
        print(arg + ':', getattr(args, arg))
    print('=' * 40)


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if args.train==True:
        main(args, device, class_list)
    if args.test==True:
        test(args, device)

