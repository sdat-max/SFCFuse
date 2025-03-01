import os
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
import random

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torch.autograd import Variable
from torchvision.utils import save_image

from args_fusion import args
# from scipy.misc import imread, imsave, imresize
import matplotlib as mpl
import torch.nn.functional as F
from torchvision import datasets, transforms

import torchvision.models as models

import torch
import torch.nn as nn
import torchvision.models as models

import torch
import torch.nn as nn
import torchvision.models as models


def t_loss(I_F, I_V, I_IR):
    block_size = 32
    margin = 0.05
    batch_size, channels, height, width = I_F.size()

    # 图像块的数量
    num_blocks = (height // block_size) * (width // block_size)

    # 计算块的l1损失
    def block_l1(image1, image2, image3):
        loss = 0
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block1 = image1[:, :, i:i + block_size, j:j + block_size]
                block2 = image2[:, :, i:i + block_size, j:j + block_size]
                block3 = image3[:, :, i:i + block_size, j:j + block_size]
                if block2.sum() > block3.sum():
                    postive_block = block2
                    negetive_block = block3
                else:
                    postive_block = block3
                    negetive_block = block2
                # l1 = F.l1_loss(block1, postive_block)
                # l2 = F.l1_loss(block1, negetive_block)
                loss += max(0, F.l1_loss(block1, postive_block) - F.l1_loss(block1, negetive_block))
        return loss

    loss = block_l1(I_F, I_V, I_IR)

    return loss


def gradients_laplacian(x):
    laplacian = [[0., 1., 0.],
                 [1., -4., 1.],
                 [0., 1., 0.]]
    laplacian_kernel = torch.FloatTensor(laplacian).unsqueeze(0).unsqueeze(0).cuda()

    edge = F.conv2d(x, laplacian_kernel, padding=1)

    return torch.abs(edge)


def gradients_sobel(x):
    kernel_x = [[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]]
    kernel_y = [[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).cuda()
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).cuda()
    sobel_x = F.conv2d(x, kernel_x, padding=1)
    sobel_y = F.conv2d(x, kernel_y, padding=1)

    grad = torch.abs(sobel_x) + torch.abs(sobel_y)
    # save_image(grad, "grad.jpg")
    return grad


def tensor_hist_equalization(tensor):
    """
    对单个通道的图像进行直方图均衡化
    Args:
        tensor (torch.Tensor): 输入Tensor，形状为 (H, W)，像素值范围为 [0, 1]
    Returns:
        equalized_tensor (torch.Tensor): 直方图均衡化后的 Tensor
    """
    # 将范围[0, 1]的浮点值转换为整数[0, 255]
    tensor = (tensor * 255).long()

    # 计算直方图
    hist = torch.histc(tensor.float(), bins=256, min=0, max=255)

    # 计算累积分布函数（CDF）
    cdf = hist.cumsum(0)
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())  # 归一化到[0, 1]区间
    cdf = (cdf * 255).long()  # 恢复到[0, 255]的像素值范围

    # 使用CDF进行插值
    equalized_tensor = cdf[tensor]

    # 恢复到[0, 1]的浮点值范围
    equalized_tensor = equalized_tensor.float() / 255.0

    return equalized_tensor


def batch_hist_equalization(batch_tensor):
    """
    对批次的四维Tensor进行直方图均衡化
    Args:
        batch_tensor (torch.Tensor): 输入Tensor，形状为 (batch_size, channels, height, width)，像素值范围为 [0, 1]
    Returns:
        equalized_batch_tensor (torch.Tensor): 直方图均衡化后的四维 Tensor
    """
    batch_size, channels, height, width = batch_tensor.size()

    # 初始化均衡化后的 batch
    equalized_batch = torch.zeros_like(batch_tensor)
    for i in range(batch_size):
        for c in range(channels):
            # 对每个通道进行直方图均衡化
            print(i)
            equalized_batch[i, c] = tensor_hist_equalization(batch_tensor[i, c])

    return equalized_batch


def gabor_kernel(ksize, sigma, gamma, lamda, alpha, psi):
    '''
    reference
      https://en.wikipedia.org/wiki/Gabor_filter
    '''

    sigma_x = sigma
    sigma_y = sigma / gamma

    ymax = xmax = ksize // 2  # 9//2
    xmin, ymin = -xmax, -ymax
    # print("xmin, ymin,xmin, ymin",xmin, ymin,ymax ,xmax)
    # X(第一个参数，横轴)的每一列一样，  Y（第二个参数，纵轴）的每一行都一样
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))  # 生成网格点坐标矩阵
    # print("y\n",y)
    # print("x\n",x)

    x_alpha = x * np.cos(alpha) + y * np.sin(alpha)
    y_alpha = -x * np.sin(alpha) + y * np.cos(alpha)
    # print("x_alpha[0][0]", x_alpha[0][0], y_alpha[0][0])
    exponent = np.exp(-.5 * (x_alpha ** 2 / sigma_x ** 2 +
                             y_alpha ** 2 / sigma_y ** 2))
    # print(exponent[0][0])
    # print(x[0],y[0])
    kernel = exponent * np.cos(2 * np.pi / lamda * x_alpha + psi)
    # print(kernel)
    # print(kernel[0][0])
    kernel = torch.from_numpy(kernel).float()
    return kernel.unsqueeze(0).unsqueeze(0).cuda()


def apply_gabor_filter(input_tensor, ksize=9, sigma=0.5, gamma=0.5, lamda=5, psi=-np.pi / 2):
    """
    对四维Tensor应用Gabor滤波器。
    """
    filters = []
    for alpha in np.arange(0, np.pi, np.pi / 4):
        # print("alpha", alpha)
        kern = gabor_kernel(ksize=ksize, sigma=sigma, gamma=gamma,
                            lamda=lamda, alpha=alpha, psi=psi)
        filters.append(kern)

    gabor_img = torch.zeros_like(input_tensor)

    # input_tensor = batch_hist_equalization(input_tensor)

    for kern in filters:
        fimg = F.conv2d(input_tensor, kern.repeat(args.batch_size, 1, 1, 1), padding=4)
        gabor_img = torch.max(gabor_img, fimg)

    p = 1.25
    gabor_img = (gabor_img - torch.min(gabor_img)) ** p
    _max = torch.max(gabor_img)
    gabor_img = gabor_img / _max
    return gabor_img



def apply_clahe(input_tensor, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    对四维Tensor（batch, channels, height, width）应用自适应直方图均衡化。
    """
    # 将 Tensor 转换为 Numpy 数组，假设输入是 (batch_size, channels, height, width)
    input_numpy = input_tensor.detach().cpu().numpy()

    # CLAHE 参数设置
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # 遍历 batch 和 channel 进行自适应直方图均衡化
    for b in range(input_numpy.shape[0]):  # batch size
        for c in range(input_numpy.shape[1]):  # channels
            # 将每个channel的图像数据归一化到 [0, 255] 范围，并确保数据为 uint8
            img = input_numpy[b, c]
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

            # 对图像进行自适应直方图均衡化
            img = clahe.apply(img)

            # 恢复到 [0, 1] 范围
            input_numpy[b, c] = img / 255.0

    # 将 Numpy 数组转换回 Tensor
    return torch.from_numpy(input_numpy).float().cuda()


def log_gabor_kernel(ksize, sigma, gamma, lamda, alpha, psi):
    '''
    生成Log-Gabor滤波器。
    '''
    sigma_x = sigma
    sigma_y = sigma / gamma

    ymax = xmax = ksize // 2
    xmin, ymin = -xmax, -ymax

    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # 将空间坐标转换为极坐标
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)

    # 调整方向
    x_alpha = x * np.cos(alpha) + y * np.sin(alpha)
    y_alpha = -x * np.sin(alpha) + y * np.cos(alpha)

    # Log-Gabor频率响应
    f0 = 1.0 / lamda  # 中心频率
    log_r = np.log(r + 1e-8)  # 对数尺度
    exponent = np.exp(-0.5 * (log_r - np.log(f0)) ** 2 / (np.log(sigma) ** 2))

    # 构建Log-Gabor滤波器
    kernel = exponent * np.cos(2 * np.pi / lamda * x_alpha + psi)

    # 将其转为Torch张量并在CUDA上计算
    kernel = torch.from_numpy(kernel).float()
    return kernel.unsqueeze(0).unsqueeze(0).cuda()


def apply_log_gabor_filter(input_tensor, ksize=9, sigma=0.5, gamma=0.5, lamda=5, psi=-np.pi / 2):
    """
    对四维Tensor应用Log-Gabor滤波器。
    """
    filters = []
    for alpha in np.arange(0, np.pi, np.pi / 4):
        kern = log_gabor_kernel(ksize=ksize, sigma=sigma, gamma=gamma,
                                lamda=lamda, alpha=alpha, psi=psi)
        filters.append(kern)

    log_gabor_img = torch.zeros_like(input_tensor)

    input_tensor = apply_clahe(input_tensor)

    for kern in filters:
        fimg = F.conv2d(input_tensor, kern.repeat(input_tensor.size(0), 1, 1, 1), padding=ksize // 2)
        log_gabor_img = torch.max(log_gabor_img, fimg)

    # 归一化和非线性处理
    p = 1.25
    log_gabor_img = (log_gabor_img - torch.min(log_gabor_img)) ** p
    _max = torch.max(log_gabor_img)
    log_gabor_img = log_gabor_img / _max
    return log_gabor_img


def lbp(image, radius=1, points=8):
    """
    计算单通道图像的局部二值模式（LBP）。
    Args:
        image (torch.Tensor): 输入的单通道图像，形状为 (H, W) 或 (1, H, W)
        radius (int): LBP 半径
        points (int): 采样点数
    Returns:
        torch.Tensor: 计算得到的 LBP 特征图，形状与输入图像相同
    """
    # 确保输入是单通道图像
    if image.ndim == 3:
        image = image.squeeze(0)  # 移除通道维度

    # 创建 LBP 特征图
    height, width = image.size()
    lbp_image = torch.zeros_like(image, dtype=torch.uint8)

    # 计算 LBP
    for p in range(points):
        # 计算角度
        theta = (p / points) * 2 * torch.pi
        # 计算邻域像素位置
        x_offset = int(radius * torch.cos(theta))
        y_offset = int(radius * torch.sin(theta))

        # 确保偏移在图像范围内
        if 0 <= x_offset < width and 0 <= y_offset < height:
            # 计算相应的 LBP 值
            shifted_image = F.pad(image, (1, 1, 1, 1), mode='replicate')
            neighbor_pixels = shifted_image[y_offset + 1:height + y_offset + 1, x_offset + 1:width + x_offset + 1]
            lbp_image += (neighbor_pixels >= image).to(torch.uint8) << p

    return lbp_image


def apply_lbp(batch_tensor, radius=1, points=8):
    """
    对批次的四维Tensor进行LBP计算。
    Args:
        batch_tensor (torch.Tensor): 输入的Tensor，形状为 (batch_size, channels, height, width)，像素值范围 [0, 1]
    Returns:
        torch.Tensor: LBP 特征图，形状为 (batch_size, channels, height, width)
    """
    batch_size, channels, height, width = batch_tensor.size()
    lbp_batch = torch.zeros((batch_size, channels, height, width), dtype=torch.uint8)

    for i in range(batch_size):
        for c in range(channels):
            lbp_batch[i, c] = lbp(batch_tensor[i, c])

    return lbp_batch



def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        elif name.endswith('.bmp'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgbimage(tensor, filename, cuda=True):
    if cuda:
        # img = tensor.clone().cpu().clamp(0, 255).numpy()
        img = tensor.cpu().clamp(0, 255).data[0].numpy()
    else:
        # img = tensor.clone().clamp(0, 255).numpy()
        img = tensor.clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def matSqrt(x):
    U, D, V = torch.svd(x)
    return U * (D.pow(0.5).diag()) * V.t()


# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    # random
    random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches


def get_image(path, height=512, width=640, mode='L'):
    if mode == 'L':
        image = Image.open(path).convert('L')
    elif mode == 'RGB':
        image = Image.open(path).convert('RGB')
    elif mode == 'YCbCr':
        img = Image.open(path).convert('YCbCr')
        image, _, _ = img.split()

    if height is not None and width is not None:
        image = image.resize((width, height), resample=Image.NEAREST)
    image = np.array(image)

    return image


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


# def get_train_images_auto(paths, height=512, width=640, mode='RGB'):
#     if isinstance(paths, str):
#         paths = [paths]
#     images = []
#     for path in paths:
#         image = get_image(path, height, width, mode=mode)
#
#         if mode == 'L':
#             image = np.reshape(image, [1, image.shape[0], image.shape[1]])
#         else:
#             image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
#         images.append(image)
#
#     images = np.stack(images, axis=0)
#     images = torch.from_numpy(images).float()
#     return images

def get_train_images_auto(paths, height=512, width=640, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    transf = transforms.ToTensor()
    for path in paths:
        image = get_image(path, height, width, mode=mode)

        # if mode == 'L':
        #     image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        # else:
        #     image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        image = transf(image)
        images.append(image)
    images = torch.stack(images)
    # images = transf(images)
    # images = torch.from_numpy(images).float()
    return images


def get_train_images(paths, height=512, width=640, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, flag)
        if flag is True:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image, [1, height, width])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


# def get_test_images(paths, height=None, width=None, mode='RGB'):
#     # ImageToTensor = transforms.Compose([transforms.ToTensor()])
#     if isinstance(paths, str):
#         paths = [paths]
#     images = []
#     transf = transforms.ToTensor()
#     for path in paths:
#         image = get_image(path, height, width, mode=mode)
#         # if mode == 'L':  # or mode == 'YCbCr':
#         #     image = np.reshape(image, [1, image.shape[0], image.shape[1]])
#         # else:
#         #     # test = ImageToTensor(image).numpy()
#         #     # shape = ImageToTensor(image).size()
#         #     image = ImageToTensor(image).float().numpy() * 255
#         images.append(image)
#         images.append(image)
#         images = np.stack(images, axis=0)
#         images = torch.from_numpy(images).float()
#     return images
def get_test_images(paths, height=None, width=None, mode='RGB'):
    # ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    transf = transforms.ToTensor()
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        # if mode == 'L':  # or mode == 'YCbCr':
        #     image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        # else:
        #     # test = ImageToTensor(image).numpy()
        #     # shape = ImageToTensor(image).size()
        #     image = ImageToTensor(image).float().numpy() * 255
        image = transf(image)
        images.append(image)
    images = torch.stack(images)
    # images = transf(images)
    # images = torch.from_numpy(images).float()
    return images


# colormap
def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00', '#FF0000',
                                                                 '#8B0000'], 256)


def save_images(path, data):
    if data.shape[2] == 1:
        data = data.reshape([data.shape[0], data.shape[1]])
    img = transforms.ToPILImage()(np.uint8(data))

    img.save(path)


def normlization(x):
    min_vals = x.amin(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
    max_vals = x.amax(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
    return (x - min_vals) / (max_vals - min_vals + 1e-5)





def save_loss_image(loss_values, name):
    # 绘制损失函数变化图并保存为图片
    plt.figure()
    plt.plot(range(len(loss_values)), loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(name)
    plt.savefig(name)
