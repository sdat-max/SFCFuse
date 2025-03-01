import numpy as np
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
# from timm.layers import to_2tuple
from torch import nn, einsum
from torch.nn import init

import torch
import torch.nn as nn

import torch.nn.functional as F
from torchvision.transforms import transforms
from torchvision.utils import save_image

import utils

import torch
import numpy as np




class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output



class Pixel_fusion(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.reflection_pad = int(np.floor(kernel_size / 2))
        self.channels = channels
        self.ca = ChannelAttention(channels * 2)
        self.conv = nn.Sequential(nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
                                  nn.ReLU(),)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.max_avg_conv = nn.Sequential(nn.Conv2d(channels * 4, channels * 2, kernel_size=3, padding=1))
        self.sg = nn.Sigmoid()

    def forward(self, vi, ir):
        # _, _, H, W = vi.shape
        x = torch.cat((vi, ir), dim=1)
        x = self.ca(x) * x
        x = self.conv(x)
        max_x, avg_x = self.max_pool(x), self.avg_pool(x)
        w = self.sg(self.max_avg_conv(torch.cat([max_x, avg_x], dim=1)))
        return x * w



class Frequency_adjust(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.amp_conv = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=1))
        # self.pha_conv = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=1),
        #                               nn.LeakyReLU())
        self.out_conv = nn.Sequential(nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1),)


    def forward(self, vi, ir):
        vi = torch.fft.fft2(vi)
        amp1 = torch.abs(vi)
        pha1 = torch.angle(vi)

        ir = torch.fft.fft2(ir)
        amp2 = torch.abs(ir)
        pha2 = torch.angle(ir)

        amp = self.amp_conv(torch.cat([amp1, amp2], dim=1))
        # pha = self.pha_conv(torch.cat([pha1, pha2], dim=1))

        real1 = amp * (torch.cos(pha1) + torch.cos(pha1))
        imag1 = amp * (torch.sin(pha1) + torch.sin(pha1))

        real2 = amp * (torch.cos(pha2) + torch.cos(pha2))
        imag2 = amp * (torch.sin(pha2) + torch.sin(pha2))

        out1 = torch.abs(torch.fft.ifft2(torch.complex(real1, imag1)))
        out2 = torch.abs(torch.fft.ifft2(torch.complex(real2, imag2)))

        out = torch.cat([out1, out2], dim=1)

        return out




class Fusion(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.freq_fusion = Frequency_adjust(channels)
        self.pixel_fusion1 = Pixel_fusion(channels, 3)
        self.out_conv = nn.Sequential(nn.Conv2d(channels * 4, channels * 2, kernel_size=3, padding=1),
                                      nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),)

    def forward(self, vi, ir):

        ff1 = self.freq_fusion(vi, ir)
        pf1 = self.pixel_fusion1(vi, ir)

        return self.out_conv(torch.cat([ff1, pf1], dim=1))
