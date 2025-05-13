import argparse
import os
import shutil
import sys

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
import logging

from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/hy-tmp/zx/dataset/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Unco_0', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
parser.add_argument('--save_result', type=int,  default=0)
parser.add_argument('--strategy', type=str, default="single",
                    help='strategy to get test value')

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return [dice * 100, jc * 100, hd95, asd]


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        # print(slice.shape)
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        # print(input.size())
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            elif FLAGS.model == "recon":
                out_main = net(input)[0]
            else:
                out_main = net(input)
            # print(out_main.size())
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            np.set_printoptions(threshold=np.inf)
            # print(label == 1)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    if "Prostate" in FLAGS.root_path:
        second_metric = first_metric
        third_metric = first_metric
    else:
        second_metric = calculate_metric_percase(prediction == 2, label == 2)
        third_metric = calculate_metric_percase(prediction == 3, label == 3)
        
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    if FLAGS.save_result:
        sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric

def test_single_volume_both(case, net1, net2, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net1.eval()
        net2.eval()

        with torch.no_grad():
            out_main1 = net1(input)
            probabilities1 = torch.softmax(out_main1, dim=1)
            out_main2 = net2(input)
            probabilities2 = torch.softmax(out_main2, dim=1)
            # 取两个模型预测概率更大的作为最终预测结果
            final_mask = torch.max(probabilities1, probabilities2)
            # 获取预测类别
            _, predicted_classes = torch.max(final_mask, dim=1)
            # 将预测结果转换为 numpy 数组
            out = predicted_classes.squeeze().cpu().numpy()
            # out = torch.argmax(out_main, dim=1).squeeze(0)
            # out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            np.set_printoptions(threshold=np.inf)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    if "Prostate" in FLAGS.root_path:
        second_metric = first_metric
        third_metric = first_metric
    else:
        second_metric = calculate_metric_percase(prediction == 2, label == 2)
        third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric

def compute_stats(metrics):
    mean = np.mean(metrics, axis=0)  # 按列计算平均值
    std = np.std(metrics, axis=0)    # 按列计算标准差
    return mean, std

def test_per_class(FLAGS, net, test_save_path):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    net.eval()

    first_metrics = []
    second_metrics = []
    third_metrics = []
    test_set_metrics = []
    iter = 1
    for case in image_list:
        # 获取单个样本的三个类的指标
        first_metric, second_metric, third_metric = test_single_volume(case, net, test_save_path, FLAGS)
        
        # 将每个类的指标转换为 numpy 数组
        first_metric = np.asarray(first_metric)
        second_metric = np.asarray(second_metric)
        third_metric = np.asarray(third_metric)
        
        # 将每个类的指标分别添加到对应的列表中
        first_metrics.append(first_metric)
        second_metrics.append(second_metric)
        third_metrics.append(third_metric)
        
        # 计算当前样本的平均指标
        dice = (first_metric[0] + second_metric[0] + third_metric[0]) / 3
        jc = (first_metric[1] + second_metric[1] + third_metric[1]) / 3
        hd = (first_metric[2] + second_metric[2] + third_metric[2]) / 3
        asd = (first_metric[3] + second_metric[3] + third_metric[3]) / 3
        
        # 将当前样本的平均指标添加到 test_set_metrics 列表中
        test_set_metrics.append([dice, jc, hd, asd])
        iter += 1

    # 将每个类的指标转换为 numpy 数组，方便后续计算
    first_metrics = np.array(first_metrics)
    second_metrics = np.array(second_metrics)
    third_metrics = np.array(third_metrics)
    test_set_metrics = np.array(test_set_metrics)

    # 计算每个类的平均值和标准差
    first_mean, first_std = compute_stats(first_metrics)
    second_mean, second_std = compute_stats(second_metrics)
    third_mean, third_std = compute_stats(third_metrics)
    overall_mean, overall_std = compute_stats(test_set_metrics)

    # 打印每个类的平均值和标准差
    logging.info("\nFirst Class Metrics:")
    logging.info("Dice: %.5f ± %.5f" % (first_mean[0], first_std[0]))
    logging.info("JC: %.5f ± %.5f" % (first_mean[1], first_std[1]))
    logging.info("HD95: %.5f ± %.5f" % (first_mean[2], first_std[2]))
    logging.info("ASD: %.5f ± %.5f" % (first_mean[3], first_std[3]))

    logging.info("\nSecond Class Metrics:")
    logging.info("Dice: %.5f ± %.5f" % (second_mean[0], second_std[0]))
    logging.info("JC: %.5f ± %.5f" % (second_mean[1], second_std[1]))
    logging.info("HD95: %.5f ± %.5f" % (second_mean[2], second_std[2]))
    logging.info("ASD: %.5f ± %.5f" % (second_mean[3], second_std[3]))

    logging.info("\nThird Class Metrics:")
    logging.info("Dice: %.5f ± %.5f" % (third_mean[0], third_std[0]))
    logging.info("JC: %.5f ± %.5f" % (third_mean[1], third_std[1]))
    logging.info("HD95: %.5f ± %.5f" % (third_mean[2], third_std[2]))
    logging.info("ASD: %.5f ± %.5f" % (third_mean[3], third_std[3]))


    # 打印总体的平均值和标准差
    logging.info("\nOverall Metrics:")
    logging.info("Dice: %.5f ± %.5f" % (overall_mean[0], overall_std[0]))
    logging.info("JC: %.5f ± %.5f" % (overall_mean[1], overall_std[1]))
    logging.info("HD95: %.5f ± %.5f" % (overall_mean[2], overall_std[2]))
    logging.info("ASD: %.5f ± %.5f" % (overall_mean[3], overall_std[3]))


def test_per_class_both(FLAGS, net, net1, test_save_path):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    net.eval()
    net1.eval()

    first_metrics = []
    second_metrics = []
    third_metrics = []
    test_set_metrics = []
    iter = 1
    for case in image_list:
        # 获取单个样本的三个类的指标
        first_metric, second_metric, third_metric = test_single_volume_both(case, net, net1, test_save_path, FLAGS)
        
        # 将每个类的指标转换为 numpy 数组
        first_metric = np.asarray(first_metric)
        second_metric = np.asarray(second_metric)
        third_metric = np.asarray(third_metric)
        
        # 将每个类的指标分别添加到对应的列表中
        first_metrics.append(first_metric)
        second_metrics.append(second_metric)
        third_metrics.append(third_metric)
        
        # 计算当前样本的平均指标
        dice = (first_metric[0] + second_metric[0] + third_metric[0]) / 3
        jc = (first_metric[1] + second_metric[1] + third_metric[1]) / 3
        hd = (first_metric[2] + second_metric[2] + third_metric[2]) / 3
        asd = (first_metric[3] + second_metric[3] + third_metric[3]) / 3
        
        # 将当前样本的平均指标添加到 test_set_metrics 列表中
        test_set_metrics.append([dice, jc, hd, asd])
        iter += 1

    # 将每个类的指标转换为 numpy 数组，方便后续计算
    first_metrics = np.array(first_metrics)
    second_metrics = np.array(second_metrics)
    third_metrics = np.array(third_metrics)
    test_set_metrics = np.array(test_set_metrics)

    # 计算每个类的平均值和标准差
    first_mean, first_std = compute_stats(first_metrics)
    second_mean, second_std = compute_stats(second_metrics)
    third_mean, third_std = compute_stats(third_metrics)
    overall_mean, overall_std = compute_stats(test_set_metrics)

    # 打印每个类的平均值和标准差
    logging.info("\nFirst Class Metrics:")
    logging.info("Dice: %.5f ± %.5f" % (first_mean[0], first_std[0]))
    logging.info("JC: %.5f ± %.5f" % (first_mean[1], first_std[1]))
    logging.info("HD95: %.5f ± %.5f" % (first_mean[2], first_std[2]))
    logging.info("ASD: %.5f ± %.5f" % (first_mean[3], first_std[3]))

    logging.info("\nSecond Class Metrics:")
    logging.info("Dice: %.5f ± %.5f" % (second_mean[0], second_std[0]))
    logging.info("JC: %.5f ± %.5f" % (second_mean[1], second_std[1]))
    logging.info("HD95: %.5f ± %.5f" % (second_mean[2], second_std[2]))
    logging.info("ASD: %.5f ± %.5f" % (second_mean[3], second_std[3]))

    logging.info("\nThird Class Metrics:")
    logging.info("Dice: %.5f ± %.5f" % (third_mean[0], third_std[0]))
    logging.info("JC: %.5f ± %.5f" % (third_mean[1], third_std[1]))
    logging.info("HD95: %.5f ± %.5f" % (third_mean[2], third_std[2]))
    logging.info("ASD: %.5f ± %.5f" % (third_mean[3], third_std[3]))


    # 打印总体的平均值和标准差
    logging.info("\nOverall Metrics:")
    logging.info("Dice: %.5f ± %.5f" % (overall_mean[0], overall_std[0]))
    logging.info("JC: %.5f ± %.5f" % (overall_mean[1], overall_std[1]))
    logging.info("HD95: %.5f ± %.5f" % (overall_mean[2], overall_std[2]))
    logging.info("ASD: %.5f ± %.5f" % (overall_mean[3], overall_std[3]))



if __name__ == '__main__':
    FLAGS = parser.parse_args()
    snapshot_path = "../model/{}_{}_labeled/{}".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    logging.basicConfig(filename=snapshot_path+"/res.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(FLAGS))
    test_save_path = "../model/{}_{}_labeled/{}_predictions/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
    net1 = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
    if FLAGS.strategy == "single":
        for i in range(1, 3):
            save_model_path = os.path.join(snapshot_path, '{}_best_model{}.pth'.format(FLAGS.model, i))
            net.load_state_dict(torch.load(save_model_path))
            logging.info("init weight from {}".format(save_model_path))
            test_per_class(FLAGS, net, test_save_path)
    else: # both
        logging.info("choose which has higher probility")
        save_model_path = os.path.join(snapshot_path, '{}_best_model1.pth'.format(FLAGS.model))
        net.load_state_dict(torch.load(save_model_path))
        save_model_path = os.path.join(snapshot_path, '{}_best_model2.pth'.format(FLAGS.model))
        net1.load_state_dict(torch.load(save_model_path))
        test_per_class_both(FLAGS, net, net1, test_save_path)

