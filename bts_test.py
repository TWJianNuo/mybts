# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

from __future__ import absolute_import, division, print_function

import os
import argparse
import time
import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from bts_dataloader import *

import errno
import matplotlib.pyplot as plt
from tqdm import tqdm

from bts_dataloader import *
from bts import BtsModel

from util import *


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name', type=str, help='model name', default='bts_nyu_v2')
parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts',
                    default='densenet161_bts')
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=80)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='nyu')
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--save_lpg', help='if set, save outputs from lpg layers', action='store_true')
parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--usesyncnorm', help='if set, save outputs from lpg layers', action='store_true')


if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

# model_dir = os.path.dirname(args.checkpoint_path)
# sys.path.append(model_dir)
#
# for key, val in vars(__import__(args.model_name)).items():
#     if key.startswith('__') and key.endswith('__'):
#         continue
#     vars()[key] = val


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def test(params):
    """Test function."""
    args.mode = 'test'
    args.num_threads = 0
    dataloader = BtsDataLoader(args, 'test')
    
    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)
    
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    num_test_samples = get_num_lines(args.filenames_file)

    with open(args.filenames_file) as f:
        lines = f.readlines()

    print('now testing {} files with {}'.format(num_test_samples, args.checkpoint_path))

    save_name = os.path.join('/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Bts_Pred', 'result_' + args.model_name)

    os.makedirs(os.path.join(save_name, 'vls'), exist_ok=True)
    os.makedirs(os.path.join(save_name, 'pred'), exist_ok=True)

    count = 0
    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader.data)):
            image = Variable(sample['image'].cuda())
            focal = Variable(sample['focal'].cuda())
            # Predict
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)

            figname = lines[count].split(' ')[0]
            comps = figname.split('/')
            figname = "{}_{}".format(comps[1], comps[4])

            fig1 = tensor2rgb(image, viewind=0)
            fig2 = tensor2disp(1 / depth_est, vmax=0.15, viewind=0)
            figcombined = np.concatenate([np.array(fig1), np.array(fig2)], axis=0)
            pil.fromarray(figcombined).save(os.path.join(save_name, 'vls',  figname))
            depth_estnp = depth_est[0,0,:,:].cpu().numpy() * 256.0
            depth_estnp = depth_estnp.astype(np.uint16)
            pil.fromarray(depth_estnp).save(os.path.join(save_name, 'pred', figname))
            count = count + 1
    return


if __name__ == '__main__':
    test(args)
