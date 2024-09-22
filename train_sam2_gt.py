import torch
import os, random, gc
import argparse
from PIL import Image
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from Datasets.synthia_dataset import synthia_dataset
from sam2.modeling.sam2_ssm import SAM2Base
from sam2.sam2_image_predictor import SAM2ImagePredictor

from utils.utils import colorEncode, save_checkpoint
from utils.compute_iou import DiceLoss, fast_hist, per_class_iu

import csv
from scipy.io import loadmat

colors = loadmat('/media/NAS/nas_70/siwoo_data/UDA_citycapes/color150.mat')['colors']
# names = {}
# with open('data/object150_info.csv') as f:
#     reader = csv.reader(f)
#     next(reader)
#     for row in reader:
#         names[int(row[0])] = row[5].split(";")[0]

def main(args, device, class_list):
    f = open(os.path.join(args.result, 'log.txt'), 'w')
    f.write('=' * 40)
    f.write('Arguments')
    f.write(str(args))
    f.write('=' * 40)


    import torch

    sam2_checkpoint = "sam2_hiera_small.pt"  # path to model weight
    from hydra import compose
    model_cfg = compose(config_name="sam2_hiera_s.yaml")

    sam2_model = SAM2Base(model_cfg.model)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.model.sam_mask_decoder.train(True)

    predictor = predictor.to(device)

    optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=1.0e-7)

    scaler = torch.cuda.amp.GradScaler()  # set mixed precision

    train_dataset = synthia_dataset(args, 'train')
    val_dataset = synthia_dataset(args, 'val')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset)

    for epoch in range(args.epochs):
        for iter, batch in enumerate(train_dataloader): # batch[0]
            with torch.cuda.amp.autocast():  # cast to mix precision
                img = batch[0][0]
                mask = batch[0][2].squeeze(1).to(device)
                input_point = None
                input_label = None
                # image, mask, input_point, input_label = read_batch(data)  # load data batch
                predictor.set_image(img)  # apply SAM image encoder to the image
                mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None,
                                                                                        mask_logits=None, normalize_coords=True)
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), boxes=None,
                                                                                         masks=None, )

                batched_mode = unnorm_coords.shape[0] > 1  # multi mask prediction
                high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in
                                     predictor._features["high_res_feats"]]
                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True, repeat_image=batched_mode, high_res_features=high_res_features, )
                prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[
                    -1])  # Upscale the masks to the original image resolution

                print(prd_masks.shape)


    #
    #
    # criterion_dice = DiceLoss()
    # criterion_ce = torch.nn.CrossEntropyLoss()
    #
    #
    #
    #
    #
    # max_miou = 0
    # total_train_loss = []
    # for epoch in range(args.epochs):
    #     if args.resume != 0:
    #         epoch += args.resume
    #
    #     os.makedirs(os.path.join(args.result, 'img', str(epoch)), exist_ok=True)
    #     model.train()
    #     train_loss = 0
    #
    #     for iter, batch in enumerate(train_dataloader): # batch[0]
    #         img = batch[0][0]
    #         if args.use_sam == True:
    #             img2 = batch[0][1]
    #             img = torch.cat((img, img2), dim=1)
    #
    #         mask = batch[0][2].squeeze(1).to(device)
    #         img_name = batch[1][0]
    #         pred = model(img.to(device))['out']
    #
    #         iou_loss = criterion_dice(pred, mask)
    #         ce_loss = criterion_ce(pred, mask)
    #         loss = iou_loss + ce_loss
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         lr_scheduler.step()
    #
    #         train_loss += loss / len(train_dataloader)
    #
    #         if (iter + 1) % args.print_fq == 0:
    #             print('{}/{} epoch, {}/{} batch, train loss: {}, ce: {}, iou: {}'.format(epoch,
    #                                                                                         args.epochs,
    #                                                                                         iter + 1,
    #                                                                                         len(train_dataloader),
    #                                                                                         loss, ce_loss,
    #                                                                                         iou_loss,
    #                                                                                         ))
    #
    #             if args.plt == True:
    #                 import matplotlib.pyplot as plt
    #
    #                 def norm(img):
    #                     return (img - np.min(img)) / (np.max(img) - np.min(img))
    #
    #                 plt.clf()
    #                 img1 = norm(batch[0][0].detach().cpu().numpy()[0].transpose(1, 2, 0))
    #                 if args.num_hq_token == 1:
    #                     plt.subplot(1, 5, 1)
    #                     plt.imshow(img1)
    #                     plt.subplot(1, 5, 2)
    #                     plt.imshow(low_res_masks.detach().cpu().numpy()[0, 0])
    #
    #                     from scipy.ndimage.morphology import binary_dilation
    #                     point = binary_dilation(point.detach().cpu().numpy()[0, 0], iterations=2)
    #                     plt.subplot(1, 5, 3)
    #                     plt.imshow(point)
    #
    #                     plt.subplot(1, 5, 4)
    #                     plt.imshow(offset_gt.detach().cpu().numpy()[0])
    #                     plt.colorbar()
    #                     plt.subplot(1, 5, 5)
    #                     plt.imshow(hq_mask.detach().cpu().numpy()[0, 1])
    #                 else:
    #                     plt.subplot(2, 4, 1)
    #                     plt.imshow(img1)
    #                     plt.subplot(2, 4, 2)
    #                     # plt.imshow(label.detach().cpu().numpy()[0])
    #                     plt.imshow(low_res_masks.detach().cpu().numpy()[0, 0])
    #
    #                     if args.num_hq_token >= 3:
    #                         plt.subplot(2, 4, 3)
    #                         # plt.imshow(low_res_masks.detach().cpu().numpy()[0,0])
    #                         plt.imshow(hq_mask.detach().cpu().numpy()[0, -1])
    #                         plt.colorbar()
    #
    #                         plt.subplot(2, 4, 4)
    #                         plt.imshow(offset_gt.detach().cpu().numpy()[0][-1])
    #                         plt.colorbar()
    #
    #                     # plt.subplot(2, 4, 3)
    #                     # plt.imshow(offset_gt.detach().cpu().numpy()[0][0]>1)
    #                     # plt.colorbar()
    #                     #
    #                     # plt.subplot(2, 4, 4)
    #                     # plt.imshow(offset_gt.detach().cpu().numpy()[0][0]<-1)
    #                     # plt.colorbar()
    #
    #                     plt.subplot(2, 4, 5)
    #                     plt.imshow(hq_mask.detach().cpu().numpy()[0, 0])
    #                     plt.colorbar()
    #
    #                     plt.subplot(2, 4, 6)
    #                     plt.imshow(hq_mask.detach().cpu().numpy()[0, 1])
    #                     plt.colorbar()
    #
    #                     plt.subplot(1, 4, 3)
    #                     # plt.imshow(point.numpy()[0][0])
    #                     plt.imshow(offset_gt.detach().cpu().numpy()[0][0])
    #                     plt.colorbar()
    #
    #                     plt.subplot(1, 4, 4)
    #                     plt.imshow(offset_gt.detach().cpu().numpy()[0][1])
    #                     plt.colorbar()
    #
    #
    #
    #                     # def colorize(ch, vmin=None, vmax=None):
    #                     #     """Will clamp value value outside the provided range to vmax and vmin."""
    #                     #     cmap = plt.get_cmap("jet")
    #                     #     ch = np.squeeze(ch.astype("float32"))
    #                     #     vmin = vmin if vmin is not None else ch.min()
    #                     #     vmax = vmax if vmax is not None else ch.max()
    #                     #     ch[ch > vmax] = vmax  # clamp value
    #                     #     ch[ch < vmin] = vmin
    #                     #     ch = (ch - vmin) / (vmax - vmin + 1.0e-16)
    #                     #     # take RGB from RGBA heat map
    #                     #     ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
    #                     #     return ch_cmap
    #                     #
    #                     # aaa = torch.sigmoid(low_res_masks).detach().cpu().numpy()[0, 0]
    #                     # aaa = (aaa-np.min(aaa))/(np.max(aaa)-np.min(aaa))
    #                     # aaa = Image.fromarray((aaa*255).astype(np.uint8))
    #                     # aaa.save(os.path.join(args.result, str(epoch), img_name+ '_binary.png'))
    #                     #
    #                     # aaa = hq_mask.detach().cpu().numpy()[0, 1]
    #                     # aaa = colorize(aaa)
    #                     # aaa = Image.fromarray(aaa)
    #                     # aaa.save(os.path.join(args.result, str(epoch), img_name+ '_h.png'))
    #                     #
    #                     # aaa = hq_mask.detach().cpu().numpy()[0, 0]
    #                     # aaa = colorize(aaa)
    #                     # aaa = Image.fromarray(aaa)
    #                     # aaa.save(os.path.join(args.result, str(epoch), img_name+ '_v.png'))
    #                     #
    #                     # binary_map, instance_map, marker = make_instance_hv(torch.sigmoid(low_res_masks)[0][0].detach().cpu().numpy(),
    #                     #                                                     hq_mask[0].detach().cpu().numpy())
    #                     # instance_map = mk_colored(instance_map) * 255
    #                     # instance_map = Image.fromarray((instance_map).astype(np.uint8))
    #                     # instance_map.save(os.path.join(args.result, str(epoch), img_name+ '_inst.png'))
    #
    #                 plt.savefig(os.path.join(args.result, 'img', str(epoch), str(iter) + 'ex.png'))
    #
    #     total_train_loss.append(train_loss)
    #     print('{} epoch, mean train loss: {}'.format(epoch, total_train_loss[-1]))
    #
    #     if epoch >= args.start_val:
    #         model.eval()
    #         hist = np.zeros((args.num_classes+1, args.num_classes+1))
    #         ave_mIOUs = []
    #
    #         with torch.no_grad():
    #             for iter, batch in enumerate(val_dataloader):
    #                 if iter == 10:
    #                     break
    #                 img = batch[0][0]
    #                 if args.use_sam == True:
    #                     img2 = batch[0][1]
    #                     img = torch.cat((img, img2), dim=1)
    #
    #                 mask = batch[0][2].squeeze(1)[0]
    #                 img_name = batch[1][0]
    #                 pred = model(img.to(device))['out']
    #                 pred = torch.argmax(pred, dim=1)
    #                 pred = pred[0].detach().cpu().numpy()
    #
    #                 hist += fast_hist(mask.numpy().flatten(), pred.flatten(), args.num_classes+1)
    #
    #                 mIoUs = per_class_iu(hist)
    #                 ave_mIOUs.append(mIoUs)
    #
    #                 pred = colorEncode(pred, colors).astype(np.uint8)
    #                 pred = Image.fromarray((pred).astype(np.uint8))
    #                 pred.save(os.path.join(args.result, 'img', str(epoch), str(img_name) + '_pred.png'))
    #
    #                 mask = colorEncode(mask.detach().cpu().numpy(), colors).astype(np.uint8)
    #                 mask = Image.fromarray((mask).astype(np.uint8))
    #                 mask.save(os.path.join(args.result, 'img', str(epoch), str(img_name) + '_mask.png'))
    #
    #             ave_mIOUs = np.mean(np.array(ave_mIOUs), axis=0)
    #             f = open(os.path.join(args.result, 'img', str(epoch), "result.txt"), 'w')
    #             f.write('***test result_mask*** class_name\t{:s}\n'.format('\t'.join(class_list)))
    #             f.write('***test result_mask*** mIOU\t{:s}\n'.format('\t'.join(map(str, ave_mIOUs.tolist()))))
    #             f.write('***test result_mask*** ave mIOU\t{:s}\n'.format(str(np.nanmean(ave_mIOUs))))
    #             f.close()
    #
    #             if max_miou < np.nanmean(ave_mIOUs):
    #                 print('save {} epoch!!--ave mIOU: {}'.format(str(epoch), np.nanmean(ave_mIOUs)))
    #                 save_checkpoint(os.path.join(args.result, 'model', 'Dice_best_model.pth'), model, epoch)
    #                 max_miou = np.nanmean(ave_mIOUs)
    #
    #             print(epoch, ': ave mIOU\t{:s} (b mIOU: {}'.format(str(np.nanmean(ave_mIOUs)), max_miou))


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
    parser.add_argument('--num_classes', default=22, type=int)
    parser.add_argument('--plt', action='store_true')
    parser.add_argument('--use_sam', action='store_true')

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    parser.add_argument('--model_type', default='deeplabv3_resnet50', help='')
    parser.add_argument('--img_dir',default='/media/NAS/nas_187/datasets/Cityscapes-Seq/leftImg8bit_sequence_trainvaltest/leftImg8bit_sequence',help='')
    parser.add_argument('--mask_dir',default='/media/NAS/nas_187/datasets/synthia/RAND_CITYSCAPES/GT/LABELS')

    parser.add_argument('--epochs', default=100, type=int, help='')
    parser.add_argument('--batch_size', type=int, default=4, help='')
    parser.add_argument('--lr', default=1e-4, type=float)

    parser.add_argument('--print_fq', default=15, type=int, help='')

    parser.add_argument('--result', default='/media/NAS/nas_70/siwoo_data/UDA_result/DeeplabV3_vis', help='')
    parser.add_argument('--test_name', default='None', type=str, help='')

    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()


    if args.result != None:
        os.makedirs(os.path.join(args.result, 'img'), exist_ok=True)
        os.makedirs(os.path.join(args.result, 'model'), exist_ok=True)

    if args.model_type == 'deeplabv3_resnet50':
        args.model_path = '/media/NAS/nas_70/siwoo_data/UDA_citycapes/best_deeplabv3_resnet50_voc_os16.pth'

    class_list = ['void', 'sky', 'building', 'road', 'sidewalk', 'fence', 'vegetation', 'pole', 'car', 'traffic sign',
                  'pedestrian', 'bicycle', 'motorcycle', 'parking-slot', 'road-work', 'traffic light', 'terrain',
                  'rider', 'truck', 'bus', 'train', 'wall', 'lanemarking']

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

