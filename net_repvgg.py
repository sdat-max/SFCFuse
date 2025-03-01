import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from args_fusion import args
import SFCFM
import utils
from repvgg import RepVGGBlock, repvgg_model_convert

def vision_features(feature_maps, img_type):
    count = 0
    for features in feature_maps:
        count += 1
        features = features.unsqueeze(0)
        for index in range(features.size(1)):
            file_name = 'feature_maps_' + img_type + '_level_' + str(count) + '_channel_' + str(index) + '.png'
            output_path = 'outputs/feature_maps/' + file_name
            map = features[:, index, :, :].view(1, 1, features.size(2), features.size(3))
            map = utils.normlization(map)
            # map = map * 255
            # save images
            save_image(map, output_path)


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is True:
            out = self.norm(out)
            out = self.relu(out)
        if self.is_last is False:
            out = self.relu(out)
            # out = utils.normlization(out)
        return out


class Repvgg_net(nn.Module):
    def __init__(self, input_nc=args.input_nc, output_nc=args.output_nc, deploy=False, use_se=False):
        super(Repvgg_net, self).__init__()
        regvppblock = RepVGGBlock
        nb_filter = [16, 32, 64, 64, 32, 16]
        stride = 1
        self.deploy = deploy
        self.use_se = use_se

        self.conv_vi = ConvLayer(input_nc, nb_filter[0], 3, stride, False)
        self.pool_vi1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_vi1 = RepVGGBlock(in_channels=nb_filter[0], out_channels=nb_filter[1], kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se)
        self.pool_vi2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_vi2 = RepVGGBlock(in_channels=nb_filter[1], out_channels=nb_filter[2], kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se)
        self.pool_vi3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_vi3 = RepVGGBlock(in_channels=nb_filter[2], out_channels=nb_filter[2], kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se)

        self.conv_ir = ConvLayer(input_nc, nb_filter[0], 3, stride, False)
        self.pool_ir1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_ir1 = RepVGGBlock(in_channels=nb_filter[0], out_channels=nb_filter[1], kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se)
        self.pool_ir2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_ir2 = RepVGGBlock(in_channels=nb_filter[1], out_channels=nb_filter[2], kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se)
        self.pool_ir3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_ir3 = RepVGGBlock(in_channels=nb_filter[2], out_channels=nb_filter[2], kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se)

        # decode
        self.decode1 = RepVGGBlock(in_channels=nb_filter[3], out_channels=nb_filter[4], kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se)

        self.decode2 = RepVGGBlock(in_channels=nb_filter[4], out_channels=nb_filter[5], kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se)

        self.conv_out = ConvLayer(nb_filter[5], output_nc, kernel_size=3, stride=1, is_last=False)
        # nb_filter = [16, 32, 64, 64, 32, 16]
        self.pf1 = SFCFM.Fusion(channels=nb_filter[3], kernel_size=3)
        self.pf2 = SFCFM.Fusion(channels=nb_filter[4], kernel_size=3)
        self.pf3 = SFCFM.Fusion(channels=nb_filter[5], kernel_size=3)

        self.r_conv2 = nn.Conv2d(in_channels=nb_filter[4] * 2, out_channels=nb_filter[4], kernel_size=3, padding=1)
        self.r_conv3 = nn.Conv2d(in_channels=nb_filter[5] * 2, out_channels=nb_filter[5], kernel_size=3, padding=1)


    def fusion(self, vi, ir):
        # vi encoder
        vi1 = self.conv_vi(vi)
        vi2 = self.encode_vi1(self.pool_vi2(vi1))
        vi3 = self.encode_vi2(self.pool_vi3(vi2))
        vi4 = self.encode_vi3(vi3)

        # ir encoder
        ir1 = self.conv_ir(ir)
        ir2 = self.encode_ir1(self.pool_ir2(ir1))
        ir3 = self.encode_ir2(self.pool_ir3(ir2))
        ir4 = self.encode_ir3(ir3)

        x_f = self.pf1(vi4, ir4)
        # tensor = x_f[0, :, :, :]
        # # 添加一个维度以符合 save_image 的输入要求 [32, 1, 512, 640]
        # tensor = tensor.unsqueeze(1)
        # save_image(tensor, "vi.jpg", nrow=8, normalize=True)

        x_d1 = x_f
        x_d1 = self.decode1(x_d1)
        x_d1 = F.interpolate(x_d1, scale_factor=2, mode='bilinear')

        x_d2 = self.r_conv2(torch.cat([x_d1, self.pf2(vi2, ir2)], dim=1))
        # tensor = x_d2[0, :, :, :]
        # # 添加一个维度以符合 save_image 的输入要求 [32, 1, 512, 640]
        # tensor = tensor.unsqueeze(1)
        # save_image(tensor, "vi.jpg", nrow=8, normalize=True)
        x_d2 = self.decode2(x_d2)
        x_d2 = F.interpolate(x_d2, scale_factor=2, mode='bilinear')

        x_d3 = self.r_conv3(torch.cat([x_d2, self.pf3(vi1, ir1)], dim=1))
        # tensor = x_d3[0, :, :, :]
        # # 添加一个维度以符合 save_image 的输入要求 [32, 1, 512, 640]
        # tensor = tensor.unsqueeze(1)
        # save_image(tensor, "xd3.jpg", nrow=8, normalize=True)

        output = self.conv_out(x_d3)

        return [output]



if __name__ == '__main__':
    densefuse_model = Repvgg_net(args.input_nc, args.output_nc, deploy=False)
    densefuse_model.load_state_dict(torch.load("models/BTSFusion.model"))
    repvgg_model_convert(densefuse_model, save_path="models/test/test1_model.model", do_copy=True)
