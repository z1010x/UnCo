import os
import h5py
import traceback
from pathlib import Path
import io_
from preprocess_utils import *
import numpy as np
import glob
# from utils1.visualize import show_graphs


def normalize(data):
    # normalized_data = (data - data.mean()) / (data.std() + 1e-10)
    normalized_data = (data - data.min()) / (data.max() - data.min())
    normalized_data = normalized_data  # * 2 - 1
    return normalized_data


def save_to_h5(img, mask, filename):
    hf = h5py.File(filename, 'w')
    hf.create_dataset('image', data=img)
    hf.create_dataset('label', data=mask)
    hf.close()

DCM_data = True


def process_case(root_path, case_folders, save_to):
    try:
        for case_folder in case_folders:   # PANCREAS_0001
            if DCM_data:  # if downloaed DCM data
                if not case_folder.is_dir():
                    return
                img = []
                for inner_folder in case_folder.iterdir():
                    # 11-24-2015-PANCREAS0001-Pancreas-18957
                    for folder in inner_folder.iterdir():
                        # Pancreas-99667
                        folders = list(folder.iterdir())
                        folders.sort()
                        # ./Pancreas-CT/PANCREAS_0001/11-24-2015-PANCREAS0001-Pancreas-18957/Pancreas-99667/1-001.dcm
                        for slice_path in folders:
                            # 1-001.dcm
                            slice, spacing, affine_pre = io_.read_nii(slice_path)
                            img.append(slice)
                img = np.concatenate(img)  # depth x H x W
                print(case_folder)
                case_idx = str(case_folder)[-4:]
                label_path = '/hy-tmp/zx/paper_exp/data/raw_dataset/TCIA_pancreas_labels-02-05-2017/label' + case_idx + '.nii.gz'
                img = img.swapaxes(2, 1).swapaxes(1, 0).swapaxes(1, 2)  # make depth last
                mask = io_.read_img(label_path)
            else:  # if downloaed nii data
                # image0001.nii.gz
                img, spacing, affine_pre = io_.read_nii(os.path.join("/hy-tmp/zx/paper_exp/data/raw_dataset/Pancreas/image", case_folder))
                print(case_folder)
                case_idx = str(case_folder).split('.')[0][-4:]
                label_path = '/hy-tmp/zx/paper_exp/data/raw_dataset/Pancreas/label/label' + case_idx + '.nii.gz'
                mask, _, _ = io_.read_nii(label_path)

            assert mask.shape == img.shape, "{}, {}".format(mask.shape, img.shape)

            # show_graphs(img[:, :, 100:116].clip(-125, 275), figsize=(20, 20)), show_graphs(mask[100:116], figsize=(20, 20))

            # resample to [1, 1, 1]
            target_spacing = (1, 1, 1)
            # change spacing of depth
            spacing = (spacing[1], spacing[1], spacing[1])
            affine_pre = io_.make_affine2(spacing)
            resampled_img, affine = resample_volume_nib(img, affine_pre, spacing, target_spacing, mask=False)
            resampled_mask, affine = resample_volume_nib(mask, affine_pre, spacing, target_spacing, mask=True)
            # resampled_img, resampled_mask = img, mask

            # clip to [-125, 275]
            min_clip, max_clip = -125, 275
            resampled_img = resampled_img.clip(min_clip, max_clip)
            resampled_img = normalize(resampled_img)

            # crop image
            bbox = get_bbox_3d(resampled_mask)
            offset = 25
            bbox = expand_bbox(resampled_img, bbox, expand_size=(offset, offset, offset), min_crop_size=(96, 96, 96))
            cropped_img = crop_img(resampled_img, bbox, min_crop_size=(96, 96, 96))
            cropped_mask = crop_img(resampled_mask, bbox, min_crop_size=(96, 96, 96))

            # show_graphs(cropped_img[100:116], figsize=(10, 10)), show_graphs(cropped_mask[100:116], figsize=(10, 10), filename='mask.png')
            save_to_h5(cropped_img, cropped_mask, save_to + case_idx + '.h5')
            print('saved : {}, resampled shape : {}, cropped shape : {}'.format(case_idx, resampled_img.shape, cropped_img.shape))
    except Exception as e:
        print(e)
        # traceback.print_tb(e)
        traceback.print_exc()


def generate_h5_data(root_path, save_path):
    paths = Path(root_path)
    paths = list(paths.iterdir())
    paths.sort()
    # print(paths)
    Path(save_path).mkdir(exist_ok=True)
    # io_.multiprocess_task(process_case, paths, cpu_num=16)
    process_case(root_path, paths, save_path)


if __name__ == '__main__':
    path_to_save_generated_data = '/hy-tmp/zx/paper_exp/data/raw_dataset/data'
    root_path = "/hy-tmp/zx/paper_exp/data/raw_dataset/Pancreas-CT"
    generate_h5_data(root_path, path_to_save_generated_data)