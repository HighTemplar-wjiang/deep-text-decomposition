import datetime
import time

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
from pathlib import Path
import argparse

import utils

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import cyclegan_networks as cycnet

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='EVAL_DECOMPOSITION')
parser.add_argument('--test_path', type=str, metavar='str',
                    help='path of input image for decomposition')
parser.add_argument('--in_size', type=int, default=256, metavar='N',
                    help='size of input image during eval')
parser.add_argument('--ckptdir', type=str, default='./checkpoints',
                    help='checkpoints dir (default: ./checkpoints)')
parser.add_argument('--net_G', type=str, default='unet_256', metavar='str',
                    help='net_G: unet_512, unet_256 or unet_128 or unet_64 (default: unet_512)')
parser.add_argument('--save_output', action='store_true', default=False,
                    help='to save the output images')
parser.add_argument('--output_dir', type=str, default='./eval_output', metavar='str',
                    help='evaluation output dir (default: ./eval_output)')
args = parser.parse_args()


def load_model(args):
    net_G = cycnet.define_G(
        input_nc=3, output_nc=6, ngf=64, netG=args.net_G, use_dropout=False, norm='none').to(device)
    print('loading the best checkpoint...')
    # checkpoint = torch.load(os.path.join(args.ckptdir, 'best_ckpt.pt'))
    # checkpoint = torch.load(os.path.join(args.ckptdir, 'last_ckpt.pt'))
    checkpoint = torch.load(os.path.join(args.ckptdir, 'ckpt_0999.pt'))
    net_G.load_state_dict(checkpoint['model_G_state_dict'])
    net_G.to(device)
    net_G.eval()

    return net_G


def run_eval(args):
    print('testing actual images...')

    test_path = args.test_path
    test_images = glob.glob(os.path.join(test_path, "*.bmp"))

    # Pad to square.
    square_transform = transforms.Compose([
        # utils.SquarePad(),
        # transforms.Resize([int(self.args.in_size / 2), int(self.args.in_size / 2)]),
        # transforms.CenterCrop([int(self.args.in_size / 2), int(self.args.in_size / 2)])
        utils.CenterPad([args.in_size, args.in_size], downsample_ratio=0.32)
    ])

    test_input_batch = []
    for test_image in test_images:
        # Read image.
        img_mix = cv2.imread(test_image, cv2.IMREAD_COLOR)
        img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)
        # img_mix = TF.resize(TF.to_pil_image(img_mix), [self.args.in_size, self.args.in_size])
        img_mix = TF.to_tensor(img_mix)
        img_mix = square_transform(img_mix)
        test_input_batch.append(img_mix)

    test_input_batch = torch.stack(test_input_batch)

    # Forward network for prediction.
    with torch.no_grad():
        out = net_G(test_input_batch.to(device))
        test_pred1 = out[:, 0:3, :, :]
        test_pred2 = out[:, 3:6, :, :]

    # Slicing for predictions.
    # G_pred1 = np.array(G_pred1.cpu().detach())
    # G_pred1 = G_pred1[0, :].transpose([1, 2, 0])
    # G_pred2 = np.array(G_pred2.cpu().detach())
    # G_pred2 = G_pred2[0, :].transpose([1, 2, 0])
    # img_mix = np.array(img_mix.cpu().detach())
    # img_mix = img_mix[0, :].transpose([1, 2, 0])

    vis_input = utils.make_numpy_grid(test_input_batch)
    vis_pred1 = utils.make_numpy_grid(test_pred1)
    vis_pred2 = utils.make_numpy_grid(test_pred2)
    vis = np.concatenate([vis_input, vis_pred1, vis_pred2], axis=0)
    vis = np.clip(vis, a_min=0.0, a_max=1.0)
    file_name = os.path.join(
        args.output_dir, Path(test_path).stem + '_' + str(time.time()) + '.jpg')
    plt.imsave(file_name, vis)


if __name__ == '__main__':
    args.test_path = "./test_data/nirs/"
    args.net_G = 'unet_128'
    args.in_size = 128
    args.ckptdir = 'checkpoints'

    # args.dataset = 'mnist'
    # args.net_G = 'unet_64'
    # args.in_size = 64
    # args.ckptdir = 'checkpoints'

    args.save_output = True

    net_G = load_model(args)
    run_eval(args)
