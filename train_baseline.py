import torch
import os, random, gc
import argparse
from PIL import Image
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from Datasets.synthia_dataset import synthia_dataset
from SS_model.deeplab_v3 import modeling

from utils.compute_iou import IOU, fast_hist, per_class_iu

def main(args, device):
    f = open(os.path.join(args.result, 'log.txt'), 'w')
    f.write('=' * 40)
    f.write('Arguments')
    f.write(str(args))
    f.write('=' * 40)

    # model = modeling.__dict__[args.model_type](num_classes=12, output_stride=8)
    # model.load_state_dict(torch.load(args.model_path)['model_state'])
    # print(model)

    import torch
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    for name, param in model.named_parameters():
        print(f"name: {name}")

    model.backbone.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.classifier[-1] = nn.Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[-1] = nn.Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))


    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=1.0e-7)

    criterion_iou = IOU()
    criterion_ce = torch.nn.CrossEntropyLoss()


    train_dataset = synthia_dataset(args, 'train')
    val_dataset = synthia_dataset(args, 'val')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset)

    max_miou = 0
    total_train_loss = []
    for epoch in range(args.epochs):
        if args.resume != 0:
            epoch += args.resume

        os.makedirs(os.path.join(args.result, 'img', str(epoch)), exist_ok=True)
        model.train()
        train_loss = 0

        for iter, batch in enumerate(train_dataloader): # batch[0]
            img = batch[0][0]
            img2 = batch[0][1]
            img = torch.cat((img, img2), dim=1)

            mask = batch[0][2].squeeze(1)
            img_name = batch[1][0]
            print(img.shape, '--')
            pred = model(img)['out'][0]

            iou_loss = criterion_iou(pred, mask)
            ce_loss = criterion_ce(pred, mask)
            loss = iou_loss + ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

                if args.plt == True:
                    import matplotlib.pyplot as plt

                    def norm(img):
                        return (img - np.min(img)) / (np.max(img) - np.min(img))

                    plt.clf()
                    img1 = norm(batch[0][0].detach().cpu().numpy()[0].transpose(1, 2, 0))
                    if args.num_hq_token == 1:
                        plt.subplot(1, 5, 1)
                        plt.imshow(img1)
                        plt.subplot(1, 5, 2)
                        plt.imshow(low_res_masks.detach().cpu().numpy()[0, 0])

                        from scipy.ndimage.morphology import binary_dilation
                        point = binary_dilation(point.detach().cpu().numpy()[0, 0], iterations=2)
                        plt.subplot(1, 5, 3)
                        plt.imshow(point)

                        plt.subplot(1, 5, 4)
                        plt.imshow(offset_gt.detach().cpu().numpy()[0])
                        plt.colorbar()
                        plt.subplot(1, 5, 5)
                        plt.imshow(hq_mask.detach().cpu().numpy()[0, 1])
                    else:
                        plt.subplot(2, 4, 1)
                        plt.imshow(img1)
                        plt.subplot(2, 4, 2)
                        # plt.imshow(label.detach().cpu().numpy()[0])
                        plt.imshow(low_res_masks.detach().cpu().numpy()[0, 0])

                        if args.num_hq_token >= 3:
                            plt.subplot(2, 4, 3)
                            # plt.imshow(low_res_masks.detach().cpu().numpy()[0,0])
                            plt.imshow(hq_mask.detach().cpu().numpy()[0, -1])
                            plt.colorbar()

                            plt.subplot(2, 4, 4)
                            plt.imshow(offset_gt.detach().cpu().numpy()[0][-1])
                            plt.colorbar()

                        # plt.subplot(2, 4, 3)
                        # plt.imshow(offset_gt.detach().cpu().numpy()[0][0]>1)
                        # plt.colorbar()
                        #
                        # plt.subplot(2, 4, 4)
                        # plt.imshow(offset_gt.detach().cpu().numpy()[0][0]<-1)
                        # plt.colorbar()

                        plt.subplot(2, 4, 5)
                        plt.imshow(hq_mask.detach().cpu().numpy()[0, 0])
                        plt.colorbar()

                        plt.subplot(2, 4, 6)
                        plt.imshow(hq_mask.detach().cpu().numpy()[0, 1])
                        plt.colorbar()

                        plt.subplot(1, 4, 3)
                        # plt.imshow(point.numpy()[0][0])
                        plt.imshow(offset_gt.detach().cpu().numpy()[0][0])
                        plt.colorbar()

                        plt.subplot(1, 4, 4)
                        plt.imshow(offset_gt.detach().cpu().numpy()[0][1])
                        plt.colorbar()



                        # def colorize(ch, vmin=None, vmax=None):
                        #     """Will clamp value value outside the provided range to vmax and vmin."""
                        #     cmap = plt.get_cmap("jet")
                        #     ch = np.squeeze(ch.astype("float32"))
                        #     vmin = vmin if vmin is not None else ch.min()
                        #     vmax = vmax if vmax is not None else ch.max()
                        #     ch[ch > vmax] = vmax  # clamp value
                        #     ch[ch < vmin] = vmin
                        #     ch = (ch - vmin) / (vmax - vmin + 1.0e-16)
                        #     # take RGB from RGBA heat map
                        #     ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
                        #     return ch_cmap
                        #
                        # aaa = torch.sigmoid(low_res_masks).detach().cpu().numpy()[0, 0]
                        # aaa = (aaa-np.min(aaa))/(np.max(aaa)-np.min(aaa))
                        # aaa = Image.fromarray((aaa*255).astype(np.uint8))
                        # aaa.save(os.path.join(args.result, str(epoch), img_name+ '_binary.png'))
                        #
                        # aaa = hq_mask.detach().cpu().numpy()[0, 1]
                        # aaa = colorize(aaa)
                        # aaa = Image.fromarray(aaa)
                        # aaa.save(os.path.join(args.result, str(epoch), img_name+ '_h.png'))
                        #
                        # aaa = hq_mask.detach().cpu().numpy()[0, 0]
                        # aaa = colorize(aaa)
                        # aaa = Image.fromarray(aaa)
                        # aaa.save(os.path.join(args.result, str(epoch), img_name+ '_v.png'))
                        #
                        # binary_map, instance_map, marker = make_instance_hv(torch.sigmoid(low_res_masks)[0][0].detach().cpu().numpy(),
                        #                                                     hq_mask[0].detach().cpu().numpy())
                        # instance_map = mk_colored(instance_map) * 255
                        # instance_map = Image.fromarray((instance_map).astype(np.uint8))
                        # instance_map.save(os.path.join(args.result, str(epoch), img_name+ '_inst.png'))

                    plt.savefig(os.path.join(args.result, 'img', str(epoch), str(iter) + 'ex.png'))

        total_train_loss.append(train_loss)
        print('{} epoch, mean train loss: {}'.format(epoch, total_train_loss[-1]))

        # if epoch % 5 == 0:
        #     save_checkpoint(os.path.join(args.result, 'model', str(epoch) + '_model.pth'), sam_model, epoch)

        if epoch >= args.start_val:
            model.eval()
            mean_iou = 0
            hist = np.zeros((12, 12))

            with torch.no_grad():
                for iter, pack in enumerate(val_dataloader):
                    img = batch[0][0]
                    img2 = batch[0][1]
                    img = torch.cat((img, img2), dim=1)
                    mask = batch[0][2].squeeze(1)
                    img_name = batch[1][0]

                    pred = model(img)

                    hist += fast_hist(mask.flatten(), pred.flatten(), 12)
                    print('{:s}: {:0.2f}'.format(img_name, 100 * np.nanmean(per_class_iu(hist))))

                    mIoUs = per_class_iu(hist)
                    print(mIoUs)



                    instance_map = mk_colored(instance_map) * 255
                    instance_map = Image.fromarray((instance_map).astype(np.uint8))
                    instance_map.save(os.path.join(args.result, 'img', str(epoch), str(img_name) + '_pred_inst.png'))

                    marker = mk_colored(marker) * 255
                    marker = Image.fromarray((marker).astype(np.uint8))
                    marker.save(os.path.join(args.result, 'img', str(epoch), str(img_name) + '_marker.png'))

                    binary_map = mk_colored(binary_map) * 255
                    binary_map = Image.fromarray((binary_map).astype(np.uint8))
                    binary_map.save(os.path.join(args.result, 'img', str(epoch), str(img_name) + '_pred.png'))

                    pred_flow_vis = Image.fromarray(pred_flow_vis.astype(np.uint8))
                    pred_flow_vis.save(os.path.join(args.result, 'img', str(epoch), str(img_name) + '_flow_vis.png'))

                    mask = mk_colored(mask[0][0].detach().cpu().numpy()) * 255
                    mask = Image.fromarray((mask).astype(np.uint8))
                    mask.save(os.path.join(args.result, 'img', str(epoch), str(img_name) + '_mask.png'))

                    del binary_map, instance_map, marker, pred_flow_vis, mask, input

                f = open(os.path.join(args.result, 'img', str(epoch), "result.txt"), 'w')
                f.write('***test result_mask*** Average- Dice\tIOU\tAJI: '
                        '\t\t{:.4f}\t{:.4f}\t{:.4f}'.format(mean_dice, mean_iou, mean_aji))
                f.close()

                if max_Dice < mean_dice:
                    print('save {} epoch!!--Dice: {}'.format(str(epoch), mean_dice))
                    save_checkpoint(os.path.join(args.result, 'model', 'Dice_best_model.pth'), sam_model, epoch)
                    max_Dice = mean_dice

                if max_Aji < mean_aji:
                    print('save {} epoch!!--Aji: {}'.format(str(epoch), mean_aji))
                    save_checkpoint(os.path.join(args.result, 'model', 'Aji_best_model.pth'), sam_model, epoch)
                    max_Aji = mean_aji

                print(epoch, ': Average- Dice\tIOU\tAJI: '
                             '\t\t{:.4f}\t{:.4f}\t{:.4f} (b Dice: {}, b Aji: {})'.format(mean_dice, mean_iou,
                                                                                         mean_aji, max_Dice, max_Aji))


def test(args, device):
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # sam_model = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    # sam_model = sam_model.to(device)

    # adapter
    if args.model_type == 'vit_h':
        encoder_mode = {'name': 'sam', 'img_size': args.img_size, 'mlp_ratio': 4, 'patch_size': 16, 'qkv_bias': True, 'use_rel_pos': True, 'window_size': 14, 'out_chans': 256, 'scale_factor': 32, 'input_type': 'fft',
                        'freq_nums': 0.25, 'prompt_type': 'highpass', 'prompt_embed_dim': 256, 'tuning_stage': 1234, 'handcrafted_tune': True, 'embedding_tune': True, 'adaptor': 'adaptor', 'embed_dim': 1280,
                        'depth': 32, 'num_heads': 16, 'global_attn_indexes': [7, 15, 23, 31]}
    elif args.model_type == 'vit_l':
        encoder_mode = {'name': 'sam', 'img_size': args.img_size, 'mlp_ratio': 4, 'patch_size': 16, 'qkv_bias': True, 'use_rel_pos': True, 'window_size': 14, 'out_chans': 256, 'scale_factor': 32, 'input_type': 'fft',
                        'freq_nums': 0.25, 'prompt_type': 'highpass', 'prompt_embed_dim': 256, 'tuning_stage': 1234, 'handcrafted_tune': True, 'embedding_tune': True, 'adaptor': 'adaptor', 'embed_dim': 1024,
                        'depth': 24, 'num_heads': 16, 'global_attn_indexes': [5, 11, 17, 23]}
    elif args.model_type == 'vit_b':
        encoder_mode = {'name': 'sam', 'img_size': args.img_size, 'mlp_ratio': 4, 'patch_size': 16, 'qkv_bias': True,
                        'use_rel_pos': True, 'window_size': 14, 'out_chans': 256, 'scale_factor': 32,
                        'input_type': 'fft',
                        'freq_nums': 0.25, 'prompt_type': 'highpass', 'prompt_embed_dim': 256, 'tuning_stage': 1234,
                        'handcrafted_tune': True, 'embedding_tune': True, 'adaptor': 'adaptor', 'embed_dim': 768,
                        'depth': 12, 'num_heads': 12, 'global_attn_indexes': [2, 5, 8, 11]}
        sam_checkpiont = 'sam_vit_b_01ec64.pth'

    sam_model = models.sam_DA.SAM(inp_size=1024, encoder_mode=encoder_mode, loss='iou', device=device)

    sam_model.make_HQ_model(model_type=args.model_type, num_token=args.num_hq_token)
    sam_model = sam_model.cuda()

    # sam_checkpoint = torch.load(os.path.join(args.model, 'Aji_best_model.pth'))
    # sam_model.load_state_dict(sam_checkpoint, strict=False)
    sam_model = load_checkpoint(sam_model, os.path.join(args.result, 'model', 'Aji_best_model.pth'))
    # sam_model = load_checkpoint(sam_model, os.path.join(args.model, 'Dice_best_model.pth'))

    test_dataseet = DA_dataset(args, 'test', use_mask=args.sup, data=(args.data1, args.data2), train_IHC=args.train_IHC)

    test_dataloader = DataLoader(test_dataseet)
    if args.test_name == "None":
        args.test_name == 'test'
    else:
        args.test_name = 'test_'+args.test_name

    os.makedirs(os.path.join(args.result, 'img',args.test_name), exist_ok=True)
    sam_model.eval()
    mean_dice, mean_iou, mean_aji = 0, 0, 0
    mean_dq, mean_sq, mean_pq = 0, 0, 0
    mean_ap1, mean_ap2, mean_ap3 = 0, 0, 0
    # if torch.distributed.get_rank() == 0:

    with torch.no_grad():
        for iter, pack in enumerate(test_dataloader):
            input = pack[0][0]
            mask = pack[0][1]
            size = 224

            img_name = pack[1][0]
            print(img_name, 'is processing....')

            output, output_offset = split_forward(sam_model, input, args.img_size, device, args.num_hq_token, size)
            binary_mask = torch.sigmoid(output).detach().cpu().numpy()

            if args.num_hq_token == 2:
                pred_flow_vis = flow_to_color(output_offset[0].detach().cpu().numpy().transpose(1, 2, 0))
                binary_map, instance_map, marker = make_instance_hv(binary_mask[0][0],
                                                                    output_offset[0].detach().cpu().numpy())
            elif args.num_hq_token == 1:
                pred_flow_vis = output_offset[0][1].detach().cpu().numpy() * 255
                binary_map, instance_map, marker = make_instance_marker(binary_mask[0][0], output_offset[0][
                    1].detach().cpu().numpy(), args.ord_th)
            else:
                bg = torch.zeros(1, 1, 1000, 1000) + args.ord_th  # K 0.15
                bg = bg.to(device)
                output_offset = torch.argmax(torch.cat([bg, output_offset], dim=1), dim=1)
                pred_flow_vis = ((output_offset[0].detach().cpu().numpy() * 255) / 9).astype(np.uint8)
                binary_map, instance_map, marker = make_instance_sonnet(binary_mask[0][0],
                                                                        output_offset[0].detach().cpu().numpy())

            if len(np.unique(binary_map)) == 1:
                dice, iou, aji = 0, 0, 0
                pq_list = [0, 0, 0]
                ap = [0, 0, 0]
            else:
                dice, iou = accuracy_object_level(instance_map, mask[0][0].detach().cpu().numpy())
                aji = AJI_fast(mask[0][0].detach().cpu().numpy(), instance_map, img_name)
                pq_list, _ = get_fast_pq(mask[0][0].detach().cpu().numpy(), instance_map) #[dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]
                # ap, _, _, _ = average_precision(mask[0][0].detach().cpu().numpy(), instance_map)

            mean_dice += dice / (len(test_dataloader))  # *len(local_rank))
            mean_iou += iou / (len(test_dataloader))  # len(local_rank))
            mean_aji += aji / (len(test_dataloader))

            mean_dq += pq_list[0] / (len(test_dataloader))  # *len(local_rank))
            mean_sq += pq_list[1] / (len(test_dataloader))  # len(local_rank))
            mean_pq += pq_list[2] / (len(test_dataloader))

            # mean_ap1 += ap[0] / (len(test_dataloader))
            # mean_ap2 += ap[1] / (len(test_dataloader))
            # mean_ap3 += ap[2] / (len(test_dataloader))

            instance_map = mk_colored(instance_map) * 255
            instance_map = Image.fromarray((instance_map).astype(np.uint8))
            instance_map.save(os.path.join(args.result, 'img', args.test_name, str(img_name) + '_pred_inst.png'))

            marker = mk_colored(marker) * 255
            marker = Image.fromarray((marker).astype(np.uint8))
            marker.save(os.path.join(args.result, 'img', args.test_name, str(img_name) + '_marker.png'))

            pred = mk_colored(binary_map) * 255
            pred = Image.fromarray((pred).astype(np.uint8))
            pred.save(os.path.join(args.result, 'img', args.test_name, str(img_name) + '_pred.png'))

            pred_flow_vis = Image.fromarray(pred_flow_vis.astype(np.uint8))
            pred_flow_vis.save(os.path.join(args.result, 'img', args.test_name, str(img_name) + '_flow_vis.png'))

            mask = mk_colored(mask[0][0].detach().cpu().numpy()) * 255
            mask = Image.fromarray((mask).astype(np.uint8))
            mask.save(os.path.join(args.result, 'img', args.test_name, str(img_name) + '_mask.png'))



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
    parser.add_argument('--plt', action='store_true')

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    parser.add_argument('--model_type', default='deeplabv3_resnet50', help='')
    parser.add_argument('--img1_dir',default='/media/NAS/nas_187/datasets/synthia/RAND_CITYSCAPES/RGB',help='')
    parser.add_argument('--img2_dir',default='/media/NAS/nas_70/siwoo_data/UDA_citycapes/synthia',help='sam result')
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
        main(args, device)
    if args.test==True:
        test(args, device)

