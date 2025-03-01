# Training a NestFuse network
# auto-encoder

import os

from torchvision.utils import save_image

import test1_image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from net_repvgg import Repvgg_net
from args_fusion import args
import pytorch_msssim
import torch.nn.functional as F
from torchvision.transforms import transforms
from loss_ssim import ssim

EPSILON = 1e-5

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)


def main():
    original_imgs_path = utils.list_images(args.dataset_ir)
    train_num = args.train_num

    original_imgs_path = original_imgs_path[:train_num]
    random.shuffle(original_imgs_path)  #
    img_flag = args.img_flag
    alpha_list = [10000]
    w_all_list = [[5.0, 5.0]]

    for w_w in w_all_list:
        w1, w2 = w_w
        for alpha in alpha_list:
            train(original_imgs_path, img_flag, alpha, w1, w2)


def train(original_imgs_path, img_flag, alpha, w1, w2):
    batch_size = args.batch_size
    # load network model
    nc = 1
    input_nc = args.input_nc
    output_nc = args.output_nc
    BTSFusion_model = Repvgg_net(input_nc, output_nc)

    if args.resume is not None:
        print('Resuming, initializing using weight from {}.'.format(args.resume))
        BTSFusion_model.load_state_dict(torch.load(args.resume))

    # print(BTSFusion_model)
    optimizer = Adam(BTSFusion_model.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()
    ssim_loss = pytorch_msssim.msssim

    if args.cuda:
        BTSFusion_model.cuda()

    tbar = trange(args.epochs)
    print('Start training.....')

    # creating save path
    temp_path_model = os.path.join(args.save_model_dir)
    temp_path_loss = os.path.join(args.save_loss_dir)
    if os.path.exists(temp_path_model) is False:
        os.mkdir(temp_path_model)

    if os.path.exists(temp_path_loss) is False:
        os.mkdir(temp_path_loss)

    temp_path_model_w = os.path.join(args.save_model_dir, str(w1))
    temp_path_loss_w = os.path.join(args.save_loss_dir, str(w1))
    if os.path.exists(temp_path_model_w) is False:
        os.mkdir(temp_path_model_w)

    if os.path.exists(temp_path_loss_w) is False:
        os.mkdir(temp_path_loss_w)

    Loss_feature = []
    Loss_ssim = []
    Loss_texture = []
    Loss_all = []
    count_loss = 0
    all_ssim_loss = 0.
    all_fea_loss = 0.
    all_texture_loss = 0.
    for e in tbar:
        print('Epoch %d.....' % e)
        # load training database
        image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)
        BTSFusion_model.train()
        BTSFusion_model.cuda()

        count = 0
        for batch in range(batches):
            image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
            img_ir = utils.get_train_images_auto(image_paths_ir, height=args.HEIGHT, width=args.WIDTH, mode='L')

            image_paths_vi = [x.replace('infrared', 'visible') for x in image_paths_ir]
            img_vi = utils.get_train_images_auto(image_paths_vi, height=args.HEIGHT, width=args.WIDTH, mode='L')


            count += 1
            optimizer.zero_grad()

            img_ir = Variable(img_ir, requires_grad=False)
            img_vi = Variable(img_vi, requires_grad=False)


            if args.cuda:
                img_ir = img_ir.cuda()
                img_vi = img_vi.cuda()

            outputs = BTSFusion_model.fusion(img_vi, img_ir)

            x_ir = Variable(img_ir.data.clone(), requires_grad=False)
            x_vi = Variable(img_vi.data.clone(), requires_grad=False)

            ######################### LOSS FUNCTION #########################
            loss1_value = 0.
            loss2_value = 0.
            loss3_value = 0.

            for output in outputs:
                # ssim loss
                grad_vi, grad_ir, grad_f = utils.gradients_sobel(x_vi), utils.gradients_sobel(x_ir), utils.gradients_sobel(output)
                weight_A = torch.sum(grad_vi) / (torch.sum(grad_vi) + torch.sum(grad_ir))
                weight_B = torch.sum(grad_ir) / (torch.sum(grad_vi) + torch.sum(grad_ir))
                ssim_loss_temp2 = weight_A * (1 - ssim(x_vi, output)) + weight_B * (1 - ssim(x_ir, output))

                intensity_loss_temp = F.l1_loss(torch.max(x_vi, x_ir), output)

                # texture loss
                grad_joint = torch.max(grad_vi, grad_ir)
                texture_loss = F.l1_loss(grad_f, grad_joint)

                loss1_value += 30 * ssim_loss_temp2
                loss2_value += 200 * intensity_loss_temp
                loss3_value += 400 * texture_loss

            loss1_value /= len(outputs)
            loss2_value /= len(outputs)
            loss3_value /= len(outputs)

            # total loss
            total_loss = loss1_value + loss2_value + loss3_value
            total_loss.backward()
            optimizer.step()

            all_ssim_loss += loss1_value.item()  #
            all_fea_loss += loss2_value.item()  #
            all_texture_loss += loss3_value.item()  #
            if (batch + 1) % args.log_interval == 0:
                mesg = "{}\t Epoch {}:\t[{}/{}]\t ssim loss: {:.6f}\t intensity loss: {:.6f}\t texture loss: {:.6f}\t total: {:.6f}".format(
                    time.ctime(), e + 1, count, batches,
                                  all_ssim_loss / args.log_interval,
                                  all_fea_loss / args.log_interval,
                                  all_texture_loss / args.log_interval,
                                  (all_fea_loss + all_ssim_loss + all_texture_loss) / args.log_interval
                )
                tbar.set_description(mesg)
                Loss_ssim.append(all_ssim_loss / args.log_interval)
                Loss_feature.append(all_fea_loss / args.log_interval)
                Loss_texture.append(all_texture_loss / args.log_interval)
                Loss_all.append((all_fea_loss + all_ssim_loss + all_texture_loss) / args.log_interval)
                count_loss = count_loss + 1
                all_ssim_loss = 0.
                all_fea_loss = 0.
                all_texture_loss = 0.

            # save model
        BTSFusion_model.eval()
        BTSFusion_model.cpu()

        # if e % 5 == 0:
        save_model_filename = "Epoch_" + str(e) + ".pth"
        save_model_path = os.path.join(args.save_model_dir, save_model_filename)
        torch.save(BTSFusion_model.state_dict(), save_model_path)

        tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)
        test1_image.main(str(e))

    BTSFusion_model.eval()
    BTSFusion_model.cpu()

    save_model_filename = "final.pth"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(BTSFusion_model.state_dict(), save_model_path)

    utils.save_loss_image(Loss_ssim, 'Loss_ssim')
    utils.save_loss_image(Loss_feature, 'Loss_feature')
    utils.save_loss_image(Loss_texture, 'Loss_texture')

    print("\nDone, trained model saved at", save_model_path)


def check_paths(args):
    try:
        if not os.path.exists(args.vgg_model_dir):
            os.makedirs(args.vgg_model_dir)
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
