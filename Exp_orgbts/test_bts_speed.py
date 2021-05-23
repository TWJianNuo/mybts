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

import time
import argparse
import datetime
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__))) # location of src

import torch
import torch.nn as nn
import torch.nn.utils as utils

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.tensorboard import SummaryWriter

import matplotlib
import matplotlib.cm
import threading
from tqdm import tqdm
import time

from Exp_orgbts.bts import BtsModeOrg
from Exp_orgbts.bts_dataloader import BtsDataLoader
from util import *

version_num = torch.__version__
version_num = ''.join(i for i in version_num if i.isdigit())
version_num = int(version_num.ljust(10, '0'))
if version_num > 1100000000:
    from torch.utils.tensorboard import SummaryWriter
else:
    from tensorboardX import SummaryWriter

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--encoder',                   type=str,   help='type of encoder, desenet121_bts, densenet161_bts, resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts',
                                                               default='densenet161_bts')
# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--data_path_odom',                 type=str,   help='path to the data', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)

# Log and save
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')

# Training)
parser.add_argument('--bts_size',                  type=int,   help='initial num_filters in bts', default=512)

# Preprocessing
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Multi-gpu training
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=0)
parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
# Online eval
parser.add_argument('--data_path',            type=str,   help='path to the data for online evaluation', required=False)
parser.add_argument('--gt_path',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
parser.add_argument('--filenames_file',       type=str,   help='path to the filenames text file for online evaluation', required=False)
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--num_threads_eval',          type=int,   default=0)
parser.add_argument('--batch_size',          type=int,   default=1)


# Integration
parser.add_argument("--inttimes",                  type=int,     default=1)
parser.add_argument("--clipvariance",              type=float,   default=5)
parser.add_argument("--maxrange",                  type=float,   default=100)
parser.add_argument("--intw",                      type=float,   default=1)

# Shape
parser.add_argument('--angw',                      type=float, default=0)
parser.add_argument('--vlossw',                    type=float, default=1)
parser.add_argument('--sclw',                      type=float, default=1e-3)

eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']

kbcroph = 352
kbcropw = 1216

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]

def block_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__

def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def online_eval(model, dataloader_eval, gpu, vlsroot):
    totnum = 0
    dr = 0
    for idx, eval_sample_batched in enumerate(dataloader_eval.data):
        with torch.no_grad():
            image = eval_sample_batched['image'].cuda(gpu, non_blocking=True)
            assert image.shape[0] == 1
            assert image.shape[2] == 192
            assert image.shape[3] == 1088
            focal = eval_sample_batched['focal'].cuda(gpu, non_blocking=True)

            st = time.time()
            _, _, _, _, pred_depth = model(image, focal)
            dr += time.time() - st
            totnum += 1
            print("%d Samples, Ave sec/frame: %f, Mem: %f Gb" % (
            totnum, dr / totnum, float(torch.cuda.memory_allocated() / 1024 / 1024 / 1024)))

    return

def compute_shape_loss(normoptizer, pred_shape, intrinsic, depth_gt):
    loss, _, _, _, _ = normoptizer.intergrationloss_ang(ang=pred_shape, intrinsic=intrinsic, depthMap=depth_gt)
    return loss

def compute_depth_loss(silog_criterion, pred_depth, depth_gt, mask=None):
    if mask is None:
        mask = depth_gt > 1.0
        mask = mask.to(torch.bool)
    loss = silog_criterion.forward(pred_depth, depth_gt, mask)
    return loss

def compute_intre(integrater, normoptizer, intrinsic, depth_gt, shape_pred, depth_pred, variance_pred, lambda_pred):
    pred_log = normoptizer.ang2log(intrinsic=intrinsic, ang=shape_pred)

    _, _, h, w = depth_gt.shape

    mask = torch.ones_like(depth_gt)
    singularnorm = normoptizer.ang2edge(ang=shape_pred, intrinsic=intrinsic)
    mask = mask * (1 - singularnorm)
    mask[:, :, 0:100, :] = 0
    mask = mask.int().contiguous()

    lateral_re = integrater.compute_lateralre(pred_log=pred_log, mask=mask, variance=variance_pred, depthin=depth_pred)
    int_re = depth_pred * (1 - lambda_pred) + lateral_re * lambda_pred
    return int_re, lateral_re, mask

def compute_int_loss(integrater, normoptizer, intrinsic, depth_gt, shape_pred, depth_pred, variance_pred, lambda_pred):
    gtselector = (depth_gt > 0).float()
    int_re, lateral_re, mask = compute_intre(integrater, normoptizer, intrinsic, depth_gt, shape_pred, depth_pred, variance_pred, lambda_pred)
    loss_lateral = torch.sum(torch.abs(lateral_re - depth_gt) * gtselector) / (torch.sum(gtselector) + 1)
    loss_int = torch.sum(torch.abs(int_re - depth_gt) * gtselector) / (torch.sum(gtselector) + 1)
    return loss_lateral, loss_int

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.distributed = False

    checkpoint = torch.load(os.path.join(args.checkpoint_path))

    # Create model
    model = BtsModeOrg(args)
    model = torch.nn.DataParallel(model)
    model.cuda()
    model.load_state_dict(checkpoint['model'])

    cudnn.benchmark = True

    dataloader_eval = BtsDataLoader(args, 'train')
    model.eval()
    online_eval(model=model, dataloader_eval=dataloader_eval, gpu=gpu, vlsroot=None)

if __name__ == '__main__':
    args = parser.parse_args()
    main_worker(0, 1, args)