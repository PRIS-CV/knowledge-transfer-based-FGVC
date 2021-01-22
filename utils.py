import numpy as np
import torch
import torchvision
from torchvision import models
# from resnet import resnet50
import torch.nn.functional as F
import torch.nn as nn
from model import *


def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1
    return float(lr / 2 * cos_out)


def load_model(model_name, dataset, pretrain=True, require_grad=True):
    print('==> Building model..')

    class_num = None

    if dataset == "StandCars":
        class_num = 196
    if dataset == "Birds2":
        class_num = 200
    if dataset == "Aircraft":
        class_num = 100

    if model_name == 'resnet50':
        net = models.resnet50(pretrained=pretrain)
        # net = resnet50(pretrained=pretrain, num_classes=class_num)
    elif model_name == "vgg16":
        net = models.vgg16_bn(pretrained=pretrain)
    # else:


    for param in net.parameters():
        param.requires_grad = require_grad

    if model_name == 'resnet50':
        net = model_bn_resnet50(net, 512, class_num)
    if model_name == 'vgg16':
        net = model_bn_vgg16(net, 4096, class_num)

    return net


def get_cam(feature):
    b, c, w, h = feature.shape

    mask = torch.zeros([b, w, h]).cuda()

    for index in range(b):
        feature_maps = feature[index]
        weights = torch.mean(torch.mean(feature_maps, dim=-2, keepdim=True), dim=-1, keepdim=True)
    
        # cam = torch.sum(weights * feature_maps, dim=0)
        # cam = torch.clamp(cam, 0)
        # cam = cam - torch.min(cam)
        # cam = cam / torch.max(cam)
        # mask[index, :, :] = cam

        mask[index, :, :] = torch.mean(weights * feature_maps, dim=0)

    return mask


def get_cam2(feature):
    b, c, w, h = feature.shape

    mask = torch.zeros([b, w, h]).cuda()

    for index in range(b):
        feature_maps = feature[index]
        weights = torch.max(torch.max(feature_maps, dim=-2, keepdim=True)[0], dim=-1, keepdim=True)[0]
    
        cam = torch.sum(weights * feature_maps, dim=0)
        cam = torch.clamp(cam, 0)
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        mask[index, :, :] = cam

    return mask


def get_gauss(feature):
    b, w, h = feature.shape

    mask = torch.zeros_like(feature).cuda()
    for index in range(b):
        mean_b = torch.mean(feature[index])
        std_b = torch.std(feature[index])
        mask[index, :, :] = (feature[index] - mean_b) / std_b
    return mask

def get_min_max(feature):
    b, w, h = feature.shape

    mask = torch.zeros_like(feature).cuda()
    for index in range(b):
        max_b = torch.max(feature[index])
        min_b = torch.min(feature[index])

        mask[index, :, :] = (feature[index] - min_b) / (max_b - min_b)  
    return mask


def FeatureLoss(fs, ft, util_param):        
    cam_s = get_cam(fs)
    gauss_s = get_gauss(cam_s)
    mask_s = torch.clamp(gauss_s, 0)
    
    cam_t = get_cam(ft)
    gauss_t = get_gauss(cam_t)
    mask_t = torch.abs(gauss_t)

    loss = torch.mean(mask_s * mask_t)
    # print('loss', loss.data)
    return loss


def KDLoss(outputs, labels, teacher_outputs):
    T = 10
    loss = - nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1),
                             F.softmax(teacher_outputs / T, dim=1)) *  T * T
    return loss