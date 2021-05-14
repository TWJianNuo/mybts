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
import sys
project_rootdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, project_rootdir)
sys.path.append('core')

import argparse
import time
import numpy as np
import cv2

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
from torchvision.transforms import ColorJitter
import copy

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
parser.add_argument('--bts_size', type=int, help='initial num_filters in bts', default=512)
parser.add_argument('--usesyncnorm', help='if set, save outputs from lpg layers', action='store_true')

parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--test_train_aug', action='store_true')

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return np.array([silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3])

def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return lines

def test(args):
    """Test function."""
    args.mode = 'test'
    args.num_threads = 0

    if not args.test_train_aug:
        args.filenames_file = os.path.join(project_rootdir, 'RAFT_tools_nyu/splits/nyudepthv2_test_files_with_gt.txt')
    else:
        args.filenames_file = os.path.join(project_rootdir, 'RAFT_tools_nyu/splits/nyudepthv2_train_files_with_gt.txt')

    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.test_train_aug:
        jitterparam = 1.5
        brightparam = 1.00
        photo_aug = ColorJitter(brightness=brightparam, contrast=brightparam, saturation=jitterparam, hue=jitterparam / 3.14)

    if not args.test_train_aug:
        evaluation_entries = get_num_lines(args.filenames_file)
    else:
        evaluation_entries = get_num_lines(args.filenames_file)
        import random
        random.shuffle(evaluation_entries)
        evaluation_entries = evaluation_entries[0:1000]

    print('now testing {} files with {}'.format(len(evaluation_entries), args.checkpoint_path))

    metrics = list()
    countnum = 0
    with torch.no_grad():
        for t_idx, entry in enumerate(tqdm(evaluation_entries)):
            torch.manual_seed(int(t_idx))
            if not args.test_train_aug:
                imgpath = os.path.join(args.data_path, entry.split(' ')[0])
            else:
                imgpath = args.data_path + entry.split(' ')[0]

            image = Image.open(imgpath)

            if args.test_train_aug:
                image_org = copy.deepcopy(image)
                image = photo_aug(image)
                image_auged = copy.deepcopy(image)
                image_vls = np.concatenate([np.array(image_org), np.array(image_auged)], axis=0)

            image = np.asarray(image, dtype=np.float64) / 255.0
            image = normalize(torch.from_numpy(image).permute([2, 0, 1])).unsqueeze(0)
            image = Variable(image.float().cuda())

            if not args.test_train_aug:
                gt_depth_path = os.path.join(args.data_path, entry.split(' ')[1])
            else:
                gt_depth_path = args.data_path + entry.split(' ')[1]

            focal = Variable(torch.from_numpy(np.array(float(entry.split(' ')[-1]))).unsqueeze(0).cuda())

            if os.path.exists(gt_depth_path):
                gt_depth = cv2.imread(gt_depth_path, -1)
                gt_depth = gt_depth.astype(np.float32) / 1000.0

                # Predict
                lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)

                pred_depth = depth_est.squeeze().cpu().numpy()
                # if args.do_kb_crop:
                #     height, width = gt_depth.shape
                #     top_margin = int(height - 352)
                #     left_margin = int((width - 1216) / 2)
                #     pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
                #     pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
                #     pred_depth = pred_depth_uncropped

                if args.test_train_aug:
                    image_vls = np.concatenate([np.array(image_vls), np.array(tensor2disp(depth_est, percentile=95, viewind=0))], axis=0)
                    Image.fromarray(image_vls).save(os.path.join('/media/shengjie/disk1/visualization/nyuv2_img_aug_vls', str(t_idx).zfill(5) + '.png'))

                pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
                pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
                pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
                pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

                valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

                eval_mask = np.zeros(valid_mask.shape)
                eval_mask[45:471, 41:601] = 1

                valid_mask = np.logical_and(valid_mask, eval_mask)

                metric = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
                metrics.append(metric)
                countnum += 1

    metrics = np.stack(metrics, axis=0)
    metrics = np.mean(metrics, axis=0)

    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
    for i in range(8):
        print('{:7.3f}, '.format(metrics[i]), end='')
    print('{:7.3f}'.format(metrics[8]))

    return


if __name__ == '__main__':
    test(args)
