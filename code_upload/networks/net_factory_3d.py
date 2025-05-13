from networks.unet_3D import unet_3D
from networks.vnet import VNet

def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2, mode = "train"):
    if net_type == "unet_3D":
        net = unet_3D(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "vnet" and mode == "train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "vnet" and mode == "test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    else:
        net = None
    return net