# test phase
import random

import torch
from torch.autograd import Variable
from torchvision.utils import save_image

# from net_shuffle import Shuffle_net
from net_repvgg import Repvgg_net
from PIL import Image
from torchvision import datasets, transforms
# from fvcore.nn import FlopCountAnalysis, parameter_count_table

# from net import DenseFuse_net
# from net_new import DenseFuse_net
import utils
from args_fusion import args
import numpy as np
import time
# import cv2
import os
import utils

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)


def load_model(path, input_nc, output_nc):
    nest_model = Repvgg_net(input_nc, output_nc, deploy=False)
    nest_model.load_state_dict(torch.load(path), strict=False)

    # print(parameter_count_table(nest_model))
    para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))

    nest_model.eval()
    nest_model.cuda()

    return nest_model


def _generate_fusion_image(model, img1, img2):
    model.eval()
    img_fusion = model.fusion(img1, img2)
    return img_fusion[0]


def run_demo(model, infrared_path, visible_path, output_path, mode, mode2):
    ir_img = utils.get_test_images(infrared_path, height=None, width=None, mode=mode)
    vis_img = utils.get_test_images(visible_path, height=None, width=None, mode=mode2)
    if args.cuda:
        ir_img = ir_img.cuda()
        vis_img = vis_img.cuda()
    ir_img = Variable(ir_img, requires_grad=False)
    vis_img = Variable(vis_img, requires_grad=False)
    img_fusion = _generate_fusion_image(model, vis_img, ir_img)

    output_path_i = output_path
    # # save images
    if args.cuda:
        img = img_fusion.cpu().float().clamp(0, 1).data[0].numpy()
    else:
        img = img_fusion.float().clamp(0, 1).data[0].numpy()

    img = img.transpose(1, 2, 0)
    if mode2 == 'YCbCr':
        xx = Image.open(visible_path).convert('YCbCr')
        y, cb, cr = xx.split()
        img = np.uint8((img*255.0).round())
        img = transforms.ToPILImage()(img)
        img = Image.merge('YCbCr', [img, cb, cr]).convert('RGB')
        img.save(output_path_i)
    else:
        utils.save_images(output_path_i, img)


def vision_features(feature_maps, img_type):
    count = 0
    for features in feature_maps:
        count += 1
        for index in range(features.size(1)):
            file_name = 'feature_maps_' + img_type + '_level_' + str(count) + '_channel_' + str(index) + '.png'
            output_path = 'outputs/feature_maps/' + file_name
            map = features[:, index, :, :].view(1, 1, features.size(2), features.size(3))
            map = map * 255
            # save images
            utils.save_image_test(map, output_path)


def main():
    # run demo
    in_c = 3

    test_path = "./images/LLVIP/visible/"

    output_path = 'outputs/LLVIP/'

    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    if in_c == 1:
        out_c = in_c
        mode = 'L'
        mode2 = 'YCbCr'
        model_path = args.model_path
    else:
        out_c = 1
        mode = 'L'
        mode2 = 'YCbCr'  # 'RGB'
        model_path = args.model_path
    start = time.time()
    with torch.no_grad():
        model = load_model(model_path, 1, 1)
        model.eval()
        test_image_path = utils.list_images(test_path)

        for i in range(len(test_image_path)):
            visible_path = test_image_path[i]
            infrared_path = test_image_path[i].replace('visible', 'infrared')
            output_path_i = output_path + test_image_path[i].split("/")[-1].split('.')[0] + ".png"

            run_demo(model, infrared_path, visible_path, output_path_i, mode, mode2)

    print('Done......')
    end = time.time()
    print(end - start, 's')

if __name__ == '__main__':
    main()
