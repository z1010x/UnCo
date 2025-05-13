import os
import cv2  # 图片处理三方库，用于对图片进行前后处理
import numpy as np  # 用于对多维数组进行计算
import SimpleITK as sitk
import h5py
from scipy.ndimage import zoom
from medpy import metric
from tqdm import tqdm
import nibabel as nib
import argparse
import shutil

def mha2jpg_2d(save_dir, mhaPath, wc=40, ws=300):
    # image = sitk.ReadImage(mhaPath)
    # img_data = sitk.GetArrayFromImage(image)
    # 加载Nifti1Image图像
    nifti_img = nib.load(mhaPath)
    img_data = nifti_img.get_fdata()
    image_name = os.path.split(mhaPath)[-1][0:2]
    # print(img_data.shape)
    channel = img_data.shape[-1]
    # low = wc - ws / 2
    # high = wc + ws / 2
    all_img = []
    # 将医疗图像中的取值范围设置在（wc - ws / 2， wc + ws / 2）之间
    # 然后归一化0-255之间并保存
    for s in range(channel):
        slicer = img_data[:, :, s]
        # slicer[slicer < low] = low
        # slicer[slicer > high] = high
        img = cv2.normalize(slicer, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(os.path.join(save_dir, image_name+'_'+str(s) + '.jpg'), img)
 
def mha2jpg(save_dir, mhaPath, wc=40, ws=300):
    # 使用SimpleITK读取数据，并使用GetArrayFromImage()函数获得图像信息
    image = sitk.ReadImage(mhaPath)
    image_name = os.path.split(mhaPath)[-1][0:18]
    img_data = sitk.GetArrayFromImage(image)
    channel = img_data.shape[0]
    low = wc - ws / 2
    high = wc + ws / 2
    all_img = []
    # 将医疗图像中的取值范围设置在（wc - ws / 2， wc + ws / 2）之间
    # 然后归一化0-255之间并保存
    for s in range(channel):
        slicer = img_data[s, :, :]
        slicer[slicer < low] = low
        slicer[slicer > high] = high
        img = cv2.normalize(slicer, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(os.path.join(save_dir, image_name+'_'+str(s) + '.jpg'), img)

def mha2jpg_resized(save_dir, mhaPath, wc=40, ws=300):
    # 使用SimpleITK读取数据，并使用GetArrayFromImage()函数获得图像信息
    image = sitk.ReadImage(mhaPath)
    image_name = os.path.split(mhaPath)[-1][0:18]
    img_data = sitk.GetArrayFromImage(image)
    channel = img_data.shape[0]
    low = wc - ws / 2
    high = wc + ws / 2
    all_img = []
    # 将医疗图像中的取值范围设置在（wc - ws / 2， wc + ws / 2）之间
    # 然后归一化0-255之间并保存
    for s in range(channel):
        slicer = img_data[s, :, :]
        slicer[slicer < low] = low
        slicer[slicer > high] = high
        slicer = cv2.resize(slicer,(256,256))
        img = cv2.normalize(slicer, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(os.path.join(save_dir, image_name+'_'+str(s) + '.jpg'), img)

def add_mask2image_binary(images_path, masks_path, masked_path):
    # Add binary masks to images
    # right ventricle myocardium Left ventricle
    # 黑色、红色、绿色、蓝色
    colors = [(0, 0, 0), (0, 0, 255),(0, 255, 0), (255, 0, 0)]
    for img_item in tqdm(os.listdir(images_path)):
        img_path = os.path.join(images_path, img_item)
        img = cv2.imread(img_path)
        mask_path = os.path.join(masks_path, img_item[:-4] + '.jpg')  # mask是.png格式的，image是.jpg格式的
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 将彩色mask以二值图像形式读取
        mask = mask // 64
        # print(np.max(mask), np.min(mask)) [0,1]
        # 将label和image拼接，并根据类别使用不同颜色进行着色
        colored_label = np.zeros_like(img)
        for i in range(4):
            colored_label[mask == i] = colors[i]
        # 将拼接后的图像与原始图像进行叠加
        masked = cv2.addWeighted(img, 0.7, colored_label, 0.3, 0)
        masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(masked_path, img_item), masked)
    print("Add binary masks to images success")


def save_files(folder_path, save_folder, args):
    # save folder is ../model/ACDC/xx_2loss_cross_unc_mseloss_14_labeled/
    image_folder = os.path.join(save_folder, "test2", "images")
    gt_folder = os.path.join(save_folder, "test2", "gts")
    pred_folder = os.path.join(save_folder,"test2", "preds")

    # 创建保存文件的文件夹
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(gt_folder, exist_ok=True)
    os.makedirs(pred_folder, exist_ok=True)

    if args.dataset == "ACDC":
        for dirpath, _, filenames in os.walk(folder_path):
            for filename in tqdm(filenames):
                file_path = os.path.join(dirpath, filename)
                if "img" in filename:
                    save_path = os.path.join(image_folder, filename)
                    # mha2jpg(image_folder, file_path)
                    mha2jpg_resized(image_folder, file_path)
                elif "gt" in filename:
                    save_path = os.path.join(gt_folder, filename)
                    mha2jpg_resized(gt_folder, file_path)
                elif "pred" in filename:
                    save_path = os.path.join(pred_folder, filename)
                    mha2jpg_resized(pred_folder, file_path)
    else:
        for dirpath, _, filenames in os.walk(folder_path):
            for filename in tqdm(filenames):
                file_path = os.path.join(dirpath, filename)
                if "img" in filename:
                    save_path = os.path.join(image_folder, filename)
                    mha2jpg_2d(image_folder, file_path) 
                elif "gt" in filename:
                    save_path = os.path.join(gt_folder, filename)
                    mha2jpg_2d(gt_folder, file_path)
                elif "pred" in filename:
                    save_path = os.path.join(pred_folder, filename)
                    mha2jpg_2d(pred_folder, file_path)
    return image_folder, gt_folder, pred_folder 
           
# visualization
if __name__ == '__main__':
    # test model 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="ACDC", help="ACDC,Pancreas")
    parser.add_argument('--model', type=str, default="unet", help="vnet,unet")
    parser.add_argument('--save_folder', type=str, default="/root/autodl-tmp/zx/paper_exp/model/ACDC/int_25.0_ext_10.0_7_labeled",
                        help='labeled data')
    args = parser.parse_args()
    save_folder = args.save_folder
    if args.dataset == "ACDC":
        if args.model == "urpc":
            folder_path = os.path.join(save_folder, "unet_urpc_predictions")
        elif args.model == "mcnet" :
            folder_path = os.path.join(save_folder, "mcnet2d_v2_predictions")
        else:
            folder_path = os.path.join(save_folder, "unet_predictions2")
    else:
        if args.model == "mcnet" :
            folder_path = os.path.join(save_folder, "mcnet3d_v2_predictions")
        else:
            folder_path = os.path.join(save_folder, "vnet_predictions")
        
    image_folder, gt_folder, pred_folder = save_files(folder_path, save_folder, args)
    print("transfer nii.gz to jpg for visulization success")
    
    # add pred/gt to image
    pred_masked_path = os.path.join(save_folder, "pred_masked_imgs2")
    gt_masked_path = os.path.join(save_folder, "gt_masked_imgs")
    # 创建保存文件的文件夹
    os.makedirs(pred_masked_path, exist_ok=True)
    os.makedirs(gt_masked_path, exist_ok=True)
    add_mask2image_binary(image_folder, gt_folder, gt_masked_path)
    add_mask2image_binary(image_folder, pred_folder, pred_masked_path)


