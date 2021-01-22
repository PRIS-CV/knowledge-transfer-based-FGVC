import torch
import matplotlib.pyplot as plt
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse
import os
from utils import *

i = 0
# resnet = models.resnet50(pretrained=True)  # 这里单独加载一个包含全连接层的resnet50模型
image = []


class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model.module._modules.items():  ##resnet50没有.feature这个特征，直接删除用就可以。
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers, use_cuda):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)
        self.cuda = use_cuda

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        return target_activations, output


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = torch.Tensor(preprocessed_img)
    return input


def show_cam_on_image(img, mask, name, save_path):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    # cam = heatmap + np.float32(img)
    cam = heatmap
    # cam = np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(save_path + "cam_{}.jpg".format(name), np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names, use_cuda)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.Tensor(torch.from_numpy(one_hot)).requires_grad_()
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()  ##features和classifier不包含，可以重新加回去试一试，会报错不包含这个对象。
        one_hot.backward(retain_graph=True)  ##这里适配我们的torch0.4及以上，我用的1.0也可以完美兼容。（variable改成graph即可）

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = resnet  # 这里同理，要的是一个完整的网络，不然最后维度会不匹配。
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model._modules[idx] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.Tensor(torch.from_numpy(one_hot))
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='/data/zhangsiqing/src/',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    # model = models.resnet50()  # 这里相对vgg19而言我们处理的不一样，这里需要删除fc层，因为后面model用到的时候会用不到fc层，只查到fc层之前的所有层数。
    # checkpoint = torch.load("/home/zhangsiqing/FGVC_DRY/Resnet_attention/model0.pth")
    # model.load_state_dict(checkpoint)
    
    # Model
    resume = True
    count = 1
    model_path = "/data/zhangsiqing/restart/Birds2_resnet50_param/model{}.pth".format(count)
    save_path = "/data/zhangsiqing/cam{}/".format(count)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if resume:
        net = torch.load(model_path)
    else:
        net = load_model(model_name='resnet50', pretrain=True, require_grad=False)

    model = net
    # del model.fc
    model_temp = model.features
    print(model)
    # modules = list(resnet.children())[:-1]
    # model = torch.nn.Sequential(*modules)

    # print(model)
    grad_cam = GradCam(model_temp, \
                       target_layer_names=["7"],
                       use_cuda=args.use_cuda)  ##这里改成layer4也很简单，我把每层name和size都打印出来了，想看哪层自己直接嵌套就可以了。（最后你会在终端看得到name的）
    x = os.walk(args.image_path)
    for root, dirs, filename in x:
        pass
        # print(type(grad_cam))
        # print(filename)
    # sorted(filename)
    for s in filename:
        image.append(cv2.imread(args.image_path + s, 1))
    # img = cv2.imread(filename, 1)
    for img in image:
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        input = preprocess_image(img)
        # print('input.size()=', input.size())
        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested index.
        target_index = None

        mask = grad_cam(input, target_index)
        # plt.imshow(mask,cmap='jet')
        # plt.show()
        # print(type(mask))
        i = i + 1
        show_cam_on_image(img, mask, i, save_path)

# gb_model = GuidedBackpropReLUModel(model = model, use_cuda=args.use_cuda)
# gb = gb_model(input, index=target_index)
# utils.save_image(torch.from_numpy(gb), 'gb.jpg')

# cam_mask = np.zeros(gb.shape)
# for i in range(0, gb.shape[0]):
#   cam_mask[i, :, :] = mask

# cam_gb = np.multiply(cam_mask, gb)
# utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')
