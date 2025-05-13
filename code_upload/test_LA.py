import os
import argparse
import torch
import sys
import logging
import torch
import re

import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label

from networks.net_factory_3d import net_factory_3d
from utils.test_3d_patch import test_all_case, calculate_metric_percase

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/hy-tmp/zx/dataset/LA', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='LA/Unco_2', help='exp_name')
parser.add_argument('--model', type=str,  default='vnet', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-procssing?')
parser.add_argument('--labelnum', type=int, default=16, help='labeled data')
parser.add_argument('--strategy', '-s', type=str, default="both",
                    help='strategy to get test value')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu 
snapshot_path = "../model/{}_{}_labeled/{}".format(FLAGS.exp, FLAGS.labelnum, FLAGS.model)
num_classes = 2


with open(FLAGS.root_path + '/test.list', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path + "/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]


def test_calculate_metric(model, test_save_path):
    model.eval()
    avg_metric = test_all_case(model, image_list, num_classes=num_classes,
                           patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                           save_result=True, test_save_path=test_save_path,
                           metric_detail=FLAGS.detail, nms=FLAGS.nms)
    return avg_metric

def test_single_case_both(model, model1, image, stride_xy, stride_z, patch_size, num_classes=1):
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
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                # print(test_patch.shape) torch.Size([1, 1, 112, 112, 80])

                
                with torch.no_grad():
                    y1 = model(test_patch)  # [1, 1, 112, 112, 80]
                    y2 = model1(test_patch)
                    y1 = F.softmax(y1, dim=1)
                    y2 = F.softmax(y2, dim=1)  
                    # 取两个模型预测概率更大的作为最终预测结果
                    # Calculate the average prediction
                    # avg_pred = (y1 + y2) / 2
                    # Calculate the confidence by taking the maximum value along the classes axis
                    confidence = torch.max(y1, y2)
                    # Get the prediction with higher confidence
                    y = torch.argmax(confidence, dim=1)


                y = y.cpu().data.numpy()
                y = y[0,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = (score_map[0]>0.5).astype(np.int64)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map

def test_all_case_both(model, model1, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, metric_detail=0):
    loader = tqdm(image_list) if not metric_detail else image_list
    total_metric = 0.0
    ith = 0
    test_set_metrics = []
    for image_path in loader:
        # id = image_path.split('/')[-2]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction, score_map = test_single_case_both(model, model1, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
            
        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
        
        test_set_metrics.append([single_metric[0] * 100, single_metric[1] * 100, single_metric[2], single_metric[3]])    
        if metric_detail:
            logging.info('%02d,\t%.5f, %.5f, %.5f, %.5f' % (ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))

        total_metric += np.asarray(single_metric)
        
        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path +  "%02d_pred_%.2f.nii.gz" % (ith, single_metric[0]))
            #nib.save(nib.Nifti1Image(score_map[0].astype(np.float32), np.eye(4)), test_save_path +  "%02d_scores.nii.gz" % ith)
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_img.nii.gz" % ith)
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_gt.nii.gz" % ith)
        ith += 1

    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))
    
    mean = np.mean(test_set_metrics, axis=0)
    std_var = np.std(test_set_metrics, axis=0)
    return mean, std_var


def test_calculate_metric_both(snapshot_path):
    model = net_factory_3d(net_type=FLAGS.model, in_chns=1, class_num=num_classes, mode="test")
    save_model_path_1 = os.path.join(snapshot_path, '{}_best_model1.pth'.format(FLAGS.model))
    model.load_state_dict(torch.load(save_model_path_1))
    print("init weight from {}".format(save_model_path_1))

    model1 = net_factory_3d(net_type=FLAGS.model, in_chns=1, class_num=num_classes, mode="test")
    save_model_path_2 = os.path.join(snapshot_path, '{}_best_model2.pth'.format(FLAGS.model))
    model1.load_state_dict(torch.load(save_model_path_2))
    print("init weight from {}".format(save_model_path_2))

    model.eval()
    model1.eval()
    avg_metric, var = test_all_case_both(model, model1, image_list, num_classes=num_classes,
                           patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                           save_result=True, test_save_path=test_save_path,
                           metric_detail=FLAGS.detail)
    return avg_metric, var

  
if __name__ == '__main__':
    snapshot_path = "../model/{}_{}_labeled/{}".format(
        FLAGS.exp, FLAGS.labelnum, FLAGS.model)
    logging.basicConfig(filename=snapshot_path+"/res.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(FLAGS))

    test_save_path = "../model/{}_{}_labeled/{}_predictions1/".format(FLAGS.exp, FLAGS.labelnum, FLAGS.model)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    net = net_factory_3d(net_type=FLAGS.model, in_chns=1, class_num=num_classes, mode="test")
    
    if FLAGS.strategy == "individual":
        # for test model1
        save_model_path_1 = os.path.join(snapshot_path, '{}_best_model1.pth'.format(FLAGS.model))
        net.load_state_dict(torch.load(save_model_path_1))
        logging.info("init weight from {}".format(save_model_path_1))
        metric1 = test_calculate_metric(net, test_save_path)
        logging.info("model1:")
        logging.info(metric1)
        logging.info("========================================")
        # for test model2
        test_save_path = "../model/{}_{}_labeled/{}_predictions2/".format(FLAGS.exp, FLAGS.labelnum, FLAGS.model)
        if not os.path.exists(test_save_path):
            os.makedirs(test_save_path)
        save_model_path_2 = os.path.join(snapshot_path, '{}_best_model2.pth'.format(FLAGS.model))
        net.load_state_dict(torch.load(save_model_path_2))
        logging.info("init weight from {}".format(save_model_path_2))
        metric2 = test_calculate_metric(net, test_save_path)
        logging.info("model2:")
        logging.info(metric2)
        logging.info("========================================")

    # for better val dice to predict
    path = os.path.join(snapshot_path, 'log.txt')
    test_save_path = "../model/{}_{}_labeled/{}_predictions/".format(FLAGS.exp, FLAGS.labelnum, FLAGS.model)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    with open(path, 'r') as file:
        content = file.read()
        pattern1 = r'model1_mean_dice\s*:\s*([\d.]+)'
        pattern2 = r'model2_mean_dice\s*:\s*([\d.]+)'
        matches1 = re.findall(pattern1, content)
        matches2 = re.findall(pattern2, content)
        file.close()
    last_match1 = matches1[-1]
    last_match2 = matches2[-1]

    if FLAGS.strategy == "val":
        # which val dice is higher
        logging.info("strategy which mean val dice is higher")
        if float(last_match1) > float(last_match2):
            save_mode_path = os.path.join(
                snapshot_path, '{}_best_model1.pth'.format(FLAGS.model))
        else:
            save_mode_path = os.path.join(
                snapshot_path, '{}_best_model2.pth'.format(FLAGS.model))
        net.load_state_dict(torch.load(save_mode_path))
        logging.info("init weight from {}".format(save_mode_path))
        metric = test_calculate_metric(net, test_save_path)

    if FLAGS.strategy == "both":
        # two model prediction and choose better for each image
        logging.info("strategy which two model prediction and choose better for each image")
        mean, std_var = test_calculate_metric_both(snapshot_path)
        logging.info(f'Mean values: {mean}')
        logging.info(f'Standard deviation values: {std_var}')
    # logging.info("test strategy is {}".format(FLAGS.strategy))
    # logging.info(metric)



