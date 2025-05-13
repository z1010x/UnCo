import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/hy-tmp/zx/dataset/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/ucnet_ws', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=0,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, # lr=0.01
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--int_consistency', type=float,
                    default=25.0, help='consistency')
parser.add_argument('--ext_consistency', type=float,
                    default=10.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--gpu_id', type=str, default="1")
parser.add_argument('--cross_unc', type=str, default="yes")
parser.add_argument('--iter_update', type=str, default="epoch")
parser.add_argument('--w_p', type=float, default=1.0)
parser.add_argument('--unc', type=str, default="both")


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def update_ema_bn_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.buffers(), model.buffers()):
        ema_param.data = ema_param.data * alpha + param.data * (1-alpha)


'''
    parameters adversarial constraint (cal the distance between mds1's student and mds2's student)
'''
def PAL(model1, model2):
    v1, v2 = None, None
    for (p1, p2) in zip(model1.parameters(), model2.parameters()):
        if v1 is None and v2 is None:
            v1, v2 = p1.view(-1), p2.view(-1)
        else:
            v1, v2 = torch.cat((v1, p2.view(-1)), 0), torch.cat((v2, p2.view(-1)), 0)  # 拼接
    pac_loss = 1.0 + (torch.matmul(v1, v2) / (torch.norm(v1) * torch.norm(v2)))  # + 1  # +1 is for a positive loss
    return pac_loss

'''
    input: 5 sample output list(after softmax) soft label
    output: mean square episode(keep size)
'''
def cross_sample_mse(output_list, mean_output):
    # output_list.view (unlabel_bs, 5, 4, 156,156);mean_output (6,4,256,256)
    se = torch.zeros_like(mean_output)
    for i in range(5):
        for j in range(i,5):
        # se += (output_list[:, i, :, :, :] - mean_output)**2 #(6,4,256,256)
            se += (output_list[:, i, :, :, :] - output_list[:, j, :, :, :])**2
        # tmp = (output_list[:, i, :, :, :] - mean_output)**2 #(6,4,256,256)
        # se += torch.mean(tmp, dim=1).unsqueeze(1) #(6,1, 256,256)
    unc_int = se/10.0
    return unc_int

def strong_aug_var(output_list, mean_output):
    se = torch.zeros_like(mean_output)
    for i in range(5):
        se += (output_list[:, i, :, :, :] - mean_output)**2 #(6,4,256,256)
    unc_int = se/5.0
    return unc_int

'''
    Include normal Mean-Teacher method, return  Net outputs(list prediction probability)
    Update model gradients, retain backward graph
'''
def train_one_step(args, model, ema_model, sample, label, epoch):
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(args.num_classes)
    # 定义颜色变换类的数据增强操作
    color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, 
                                          saturation=0.4, hue=0.1)
    # 定义随机数据增强操作,当p=1时，表示始终应用给定的数据增强操作。
    random_color_jitter_student = transforms.RandomApply([color_jitter], p=0.8)
    random_color_jitter_teacher = transforms.RandomApply([color_jitter], p=1.0)
    
    label_student_weak = sample[:args.labeled_bs]
    unlabel_teacher_weak = sample[args.labeled_bs:]
    # unlabel_student_strong = sample[args.labeled_bs:]
    unlabel_student_strong = random_color_jitter_student(sample[args.labeled_bs:])

    # origin_output = model(sample)
    # for label data just keep weak aug
    output_label_student_weak = model(label_student_weak)
    softmax_label_student_weak = torch.softmax(output_label_student_weak, dim=1)
    # for unlabel data apply strong aug
    output_unlabel_student_strong = model(unlabel_student_strong)
    softmax_unlabel_student_strong = torch.softmax(output_unlabel_student_strong, dim=1)
    # origin_output = torch.cat((origin_output_l, origin_output_u), dim=0)
    # origin_output_soft = torch.softmax(origin_output, dim=1)
    # ema_inputs = sample[args.labeled_bs:]
    with torch.no_grad():
        # ema_model.eval()
        output_unlabel_teacher_weak = ema_model(unlabel_teacher_weak)   #(6,4,h,w)
        softmax_unlabel_teacher_weak = torch.softmax(output_unlabel_teacher_weak, dim=1)
        # 应用随机数据增强操作生成五个不同的样
        unlabel_teacher_strongs = random_color_jitter_teacher(unlabel_teacher_weak).unsqueeze(1)
        for _ in range(4):
            temp_unlabel_teacher_strong = random_color_jitter_teacher(unlabel_teacher_weak).unsqueeze(1)
            unlabel_teacher_strongs = torch.cat(
                (unlabel_teacher_strongs, temp_unlabel_teacher_strong), dim=1
            )
        output_unlabel_teacher_strongs = ema_model(unlabel_teacher_strongs.view(args.labeled_bs*5, 1, args.patch_size[0], args.patch_size[1]))
        softmax_unlabel_teacher_strongs = torch.softmax(output_unlabel_teacher_strongs, dim=1)
        softmax_unlabel_teacher_strongs = softmax_unlabel_teacher_strongs.view(args.labeled_bs, 5, args.num_classes, args.patch_size[0], args.patch_size[1])

    label_loss_ce = ce_loss(output_label_student_weak, label[:args.labeled_bs].long())
    label_loss_dice = dice_loss(softmax_label_student_weak, label[:args.labeled_bs].unsqueeze(1))
    label_supervised_loss = label_loss_ce + label_loss_dice
    # unlabel_consistency_loss = torch.mean(((softmax_unlabel_student_strong - softmax_unlabel_teacher_weak)**2).sum(1))
    # mseloss = torch.nn.MSELoss(reduction='mean')
    # unlabel_consistency_loss = mseloss(softmax_unlabel_student_strong, softmax_unlabel_teacher_weak)
    unlabel_consistency_loss = torch.mean((softmax_unlabel_student_strong - softmax_unlabel_teacher_weak)**2)

    # pseudo_outputs = torch.argmax(ema_origin_output_soft, dim=1, keepdim=False)
    # consistency_loss = ce_loss(origin_output[args.labeled_bs:], pseudo_outputs[args.labeled_bs:].long()) \
    #     + dice_loss(origin_output_soft[args.labeled_bs:], pseudo_outputs[args.labeled_bs:].unsqueeze(1)) # both U
    return ([output_unlabel_student_strong, softmax_unlabel_student_strong], 
            [output_unlabel_teacher_weak, softmax_unlabel_teacher_weak],
            [output_unlabel_teacher_strongs, softmax_unlabel_teacher_strongs], 
            label_supervised_loss, unlabel_consistency_loss)
    
def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    iter_num = 0
    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        model.retain_graph=True
        return model

    model1 = create_model()
    ema_model1 = create_model(ema=True)

    model2 = create_model()
    ema_model2 = create_model(ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,num_workers=1)

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                        momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                        momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(args.num_classes)

    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    model1.train()
    model2.train()
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            # for each learner
            """
            return [output_unlabel_student_strong, softmax_unlabel_student_strong], 
                    [output_unlabel_teacher_weak, softmax_unlabel_teacher_weak],
                    [output_unlabel_teacher_strongs, softmax_unlabel_teacher_strongs], 
                    label_supervised_loss, unlabel_consistency_loss
            """
            output_unlabel_student_strong1, output_unlabel_teacher_weak1, output_unlabel_teacher_strongs1, \
                label_supervised_loss1, unlabel_consistency_loss1 = train_one_step(args, model1, ema_model1, volume_batch, label_batch, epoch_num)
            output_unlabel_student_strong2, output_unlabel_teacher_weak2, output_unlabel_teacher_strongs2, \
                label_supervised_loss2, unlabel_consistency_loss2 = train_one_step(args, model2, ema_model2, volume_batch, label_batch, epoch_num)
            
            with torch.no_grad():
                # ema_model1.eval()
                # ema_model2.eval()

                # avg aug samples two teacher keep cons view (unlabel_bs, 5, 4, 256,256)
                mean_1 = torch.mean(output_unlabel_teacher_strongs1[1], dim=1)
                mean_2 = torch.mean(output_unlabel_teacher_strongs2[1], dim=1)
                unc_ext_strong = (mean_1 - mean_2)**2
                # unc_ext_strong = unc_ext_strong.mean(1)

                # each teacher keep cons among 5 augs sample from same image
                unc_int_1 = strong_aug_var(output_unlabel_teacher_strongs1[1], mean_1) #[]_5  [6,4,h,w]
                unc_int_2 = strong_aug_var(output_unlabel_teacher_strongs2[1], mean_2)
                # unc_int_1 = unc_int_1.mean(1)
                # unc_int_2 = unc_int_2.mean(1)

                # raw sample two teacher keep cons
                unc_ext_weak = (output_unlabel_teacher_weak1[1] - output_unlabel_teacher_weak2[1])**2
                # unc_ext_weak = unc_ext_weak.mean(1)

                # # exploit uncertainty to refine pseudo label [6,1,256,256]
                unc_total_1 = (unc_ext_strong + unc_int_1 + unc_ext_weak) / 3
                unc_total_2 = (unc_ext_strong + unc_int_2 + unc_ext_weak) / 3

                # average weighet by exp [6,1,256,256]
                loss_weight_1 = torch.exp(-1.0 * unc_total_2)
                loss_weight_2 = torch.exp(-1.0 * unc_total_1)

            # unlabel_consistency_loss = torch.mean(((softmax_unlabel_student_strong - softmax_unlabel_teacher_weak)**2).sum(1))
            consistency_dist1 = ((output_unlabel_student_strong1[1] - output_unlabel_teacher_weak2[1]) ** 2)
            seg1_loss_sum = (consistency_dist1 * loss_weight_1).sum(-1).sum(-1).sum(-1)
            seg1_loss_deno = loss_weight_1.sum(-1).sum(-1).sum(-1) + 1e-16
            seg1_loss = (seg1_loss_sum / seg1_loss_deno).mean()
           
            
            consistency_dist2 = ((output_unlabel_student_strong2[1] - output_unlabel_teacher_weak1[1]) ** 2)
            seg2_loss_sum = (consistency_dist2 * loss_weight_2).sum(-1).sum(-1).sum(-1)
            seg2_loss_deno = loss_weight_2.sum(-1).sum(-1).sum(-1) + 1e-16
            seg2_loss = (seg2_loss_sum / seg2_loss_deno).mean()
            

            # parameter adversarial loss
            pal_loss = PAL(model1, model2)
            if args.iter_update == "iters":
                w_int = args.int_consistency * ramps.sigmoid_rampup(iter_num, max_iterations)
                w_ext = args.ext_consistency * ramps.sigmoid_rampup(iter_num, max_iterations)
            else:
                w_int = args.int_consistency * ramps.sigmoid_rampup(epoch_num, max_epoch)
                w_ext = args.ext_consistency * ramps.sigmoid_rampup(epoch_num, max_epoch)
            ## have a try iter update
            w_p = args.w_p
            loss1 = label_supervised_loss1 + w_int * unlabel_consistency_loss1 + w_ext * seg1_loss + w_p * pal_loss
            loss2 = label_supervised_loss2 + w_int * unlabel_consistency_loss2 + w_ext * seg2_loss + w_p * pal_loss

            logging.info("epoch:    %s\t iters:    %s\t    advloss: %s\t", epoch_num, iter_num, pal_loss.item())
            logging.info("intloss1: %s\t intloss2: %s\t", unlabel_consistency_loss1.item(), unlabel_consistency_loss2.item())
            logging.info("extloss1: %s\t extloss2: %s\t", seg1_loss.item(),seg2_loss.item())
            logging.info("suploss1: %s\t suploss2: %s\t", label_supervised_loss1.item(), label_supervised_loss2.item())
            logging.info("loss1:    %s\t loss2:    %s\t", loss1.item(), loss2.item())
            
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            
            loss1.backward(retain_graph=True)
            optimizer1.step()
            update_ema_variables(model1, ema_model1, args.ema_decay, iter_num)
            update_ema_bn_variables(model1, ema_model1, args.ema_decay, iter_num)
             
            loss2.backward()
            optimizer2.step()
            update_ema_variables(model2, ema_model2, args.ema_decay, iter_num)
            update_ema_bn_variables(model2, ema_model2, args.ema_decay, iter_num)

            iter_num = iter_num + 1
            lr = base_lr * (1 - iter_num / max_iterations) ** 0.9
            optimizer1.param_groups[0]["lr"] = lr
            optimizer2.param_groups[0]["lr"] = lr

            writer.add_scalar('lr', lr, iter_num)
            writer.add_scalar(
                'unsupervised_weight/unsupervised_weight', w_int, iter_num)
            writer.add_scalar('loss/total_loss1', loss1, iter_num)
            writer.add_scalar('loss/total_loss2', loss2, iter_num)
            writer.add_scalar('loss/label_supervised_loss1', label_supervised_loss1, iter_num)
            writer.add_scalar('loss/label_supervised_loss2', label_supervised_loss2, iter_num)
            writer.add_scalar('loss/consistency_loss1', unlabel_consistency_loss1, iter_num)
            writer.add_scalar('loss/consistency_loss2', unlabel_consistency_loss2, iter_num)
            writer.add_scalar('loss/cps_loss1', seg1_loss, iter_num)
            writer.add_scalar('loss/cps_loss2', seg2_loss, iter_num)
            writer.add_scalar('loss/pal_loss', pal_loss, iter_num)

            if iter_num > 2000 and iter_num % 200 == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
                performance1 = np.mean(metric_list, axis=0)[0]
                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95', mean_hd951, iter_num)
        
                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_best = os.path.join(snapshot_path,
                                                '{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_best)
                    save_best_ema = os.path.join(snapshot_path,
                                                '{}_best_ema_model1.pth'.format(args.model))
                    torch.save(ema_model1.state_dict(), save_best_ema)
                    logging.info('the best iteration of model1 is changed, now is {} and best performance1 is {}'.format(iter_num, round(best_performance1, 4)))
                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model1.train()

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
                performance2 = np.mean(metric_list, axis=0)[0]
                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95', mean_hd952, iter_num)
                
                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_best = os.path.join(snapshot_path,
                                                '{}_best_model2.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_best)
                    save_best_ema = os.path.join(snapshot_path,
                                                '{}_best_ema_model2.pth'.format(args.model))
                    torch.save(ema_model2.state_dict(), save_best_ema)
                    logging.info('the best iteration of model2 is changed, now is {} and best performance2 is {}'.format(iter_num, round(best_performance2, 4)))
                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model2.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    torch.manual_seed(args.seed)
    torch.random.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        os.remove(snapshot_path + '/code')
    shutil.copy('./'+ sys.argv[0], snapshot_path + '/'+ sys.argv[0])

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)

    # for test
    os.system("python test_ACDC_detail.py --exp {} --labeled_num {}".format(args.exp, args.labeled_num))
