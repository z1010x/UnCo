import argparse
import os
import shutil
from glob import glob
import sys
import logging
import torch
from networks.unet_3D import unet_3D
from test_3D_util import test_all_case
import math
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
from medpy import metric
from skimage.measure import label
from tqdm import tqdm
import re

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/hy-tmp/zx/dataset/BraTS2019', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='BraTS2019/Unco_2', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_3D', help='model_name')
parser.add_argument('--labeled_num', type=int, default=25,
                    help='labeled data')
parser.add_argument('--strategy', type=str, default="both",
                    help='strategy to get test value')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() == 0:
        dice, jc, hd95, asd = 0, 0, 0, 0
    else:  
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        asd = metric.binary.asd(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd

def test_single_case_both(net, net1, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = net(test_patch)  # [1, 2, 96, 96, 96]
                    y1 = torch.softmax(y1, dim=1)
                    y2 = net1(test_patch)
                    y2 = torch.softmax(y2, dim=1)
                    # 取两个模型预测概率更大的作为最终预测结果
                    y = torch.max(y1, y2)
                    # Get the prediction with higher confidence
                    # y = torch.argmax(confidence, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                # print(y.shape)
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map



def test_all_case_both(net, net1, base_dir, method="unet_3D", test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24, test_save_path=None):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/data/{}.h5".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes-1, 4))
    print("Testing begin")
    test_set_metric = []
    with open(test_save_path + "/{}.txt".format(method), "a") as f:
        for image_path in tqdm(image_list):
            ids = image_path.split("/")[-1].replace(".h5", "")
            h5f = h5py.File(image_path, 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]
            prediction = test_single_case_both(
                net, net1, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
            metric = calculate_metric_percase(prediction == 1, label == 1)
            test_set_metric.append(metric)

            print(metric)
            total_metric[0, :] += metric
            f.writelines("{},{},{},{},{}\n".format(
                ids, metric[0], metric[1], metric[2], metric[3]))

            pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
            pred_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(pred_itk, test_save_path +
                            "/{}_pred.nii.gz".format(ids))

            img_itk = sitk.GetImageFromArray(image)
            img_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(img_itk, test_save_path +
                            "/{}_img.nii.gz".format(ids))

            lab_itk = sitk.GetImageFromArray(label.astype(np.uint8))
            lab_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(lab_itk, test_save_path +
                            "/{}_lab.nii.gz".format(ids))
        f.writelines("Mean metrics,{},{},{},{}".format(total_metric[0, 0] / len(image_list), total_metric[0, 1] / len(
            image_list), total_metric[0, 2] / len(image_list), total_metric[0, 3] / len(image_list)))
    f.close()
    print("Testing end")
    mean = np.mean(test_set_metric, axis=0)
    std_var = np.std(test_set_metric, axis=0)
    return mean, std_var


def Inference_both(args):
    num_classes = 2
    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    test_save_path = "../model/{}_{}_labeled/{}/predictions/".format(
        args.exp, args.labeled_num, args.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = unet_3D(n_classes=num_classes, in_channels=1).cuda()
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model1.pth'.format(args.model))
    # save_mode_path = "/root/autodl-tmp/zx/paper_exp/model/unet_3D_best_model1.pth"
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))

    net1 = unet_3D(n_classes=num_classes, in_channels=1).cuda()
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model2.pth'.format(args.model))
    # save_mode_path = "/root/autodl-tmp/zx/paper_exp/model/unet_3D_best_model2.pth"
    net1.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    net1.eval()
    mean, std_var = test_all_case_both(net, net1, base_dir=args.root_path, method=args.model, test_list="test.txt", num_classes=num_classes,
                               patch_size=(96, 96, 96), stride_xy=64, stride_z=64, test_save_path=test_save_path)
    return mean, std_var

def Inference1(args):
    num_classes = 2
    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    test_save_path = "../model/{}_{}_labeled/{}/predictions_1/".format(
        args.exp, args.labeled_num, args.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = unet_3D(n_classes=num_classes, in_channels=1).cuda()
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model1.pth'.format(args.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric = test_all_case(net, base_dir=args.root_path, method=args.model, test_list="test.txt", num_classes=num_classes,
                               patch_size=(96, 96, 96), stride_xy=64, stride_z=64, test_save_path=test_save_path)
    return avg_metric


def Inference2(args):
    num_classes = 2
    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    test_save_path = "../model/{}_{}_labeled/{}/predictions_2/".format(
        args.exp, args.labeled_num, args.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = unet_3D(n_classes=num_classes, in_channels=1).cuda()
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model2.pth'.format(args.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric = test_all_case(net, base_dir=args.root_path, method=args.model, test_list="test.txt", num_classes=num_classes,
                               patch_size=(96, 96, 96), stride_xy=64, stride_z=64, test_save_path=test_save_path)
    return avg_metric

if __name__ == '__main__':
    args = parser.parse_args()
    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    logging.basicConfig(filename=snapshot_path+"/res.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    logging.info("test strategy is {}".format(args.strategy))
    if args.strategy == "individual":
        logging.info("strategy individual mean val dice")
        metric1 = Inference1(args)
        logging.info("model1:")
        logging.info(metric1)
        logging.info("========================================")
        metric2 = Inference2(args)
        logging.info("model2:")
        logging.info(metric2)
    
    # for better val dice to predict
    path = os.path.join(snapshot_path, 'log.txt')
    test_save_path = "../model/{}_{}_labeled/{}_predictions/".format(args.exp, args.labeled_num, args.model)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    
    if args.strategy == "val":
        with open(path, 'r') as file:
            content = file.read()
            pattern1 = r'model1_mean_dice\s*:\s*([\d.]+)'
            pattern2 = r'model2_mean_dice\s*:\s*([\d.]+)'
            matches1 = re.findall(pattern1, content)
            matches2 = re.findall(pattern2, content)
            file.close()
        last_match1 = matches1[-1]
        last_match2 = matches2[-1]
        # which val dice is higher
        logging.info("strategy which mean val dice is higher")
        if float(last_match1) > float(last_match2):
            metric = Inference1(args)
        else:
            metric = Inference2(args)
        logging.info(metric)

    if args.strategy == "both":
        # two model prediction and choose better for each image
        logging.info("strategy which two model prediction and choose better for each image")
        mean, std_var = Inference_both(args)
        logging.info(f'Mean values: {mean}')
        logging.info(f'Standard deviation values: {std_var}')

# [[ 0.82707957  0.22336143 15.05108507  3.78368749]] Fully_supervised 25 brats2019