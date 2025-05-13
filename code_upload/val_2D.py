import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() == 0:
        dice = 0
        hd95 = 0
    else:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
    return dice, hd95


# ([1, 3, 256, 256]) ([1, 256, 256])
def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_pancreas(image, label, net, classes, patch_size=[512, 512]):
    net.eval()  # bchw
    # print(label.shape)  #[1, 512, 512])
    # print(image.shape)  #[1,1,512,512]
    label = label.cpu().detach().numpy()
    with torch.no_grad():
        # image = (1,3,512,512)
        # out = (1,2,h,w)
        prediction = torch.argmax(torch.softmax(
            net(image), dim=1), dim=1)
        prediction = prediction.cpu().detach().numpy()
        # print(prediction.shape) #[1,512,512]
    
    metric = calculate_metric_percase(prediction, label)
    return metric

   
