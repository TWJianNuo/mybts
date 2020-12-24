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

from Exp_bts_Integration.btsnet import BtsSDModel
from Exp_bts_Integration.bts_dataloader import BtsDataLoader
from integrationModule import CRFIntegrationModule
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

parser.add_argument('--encoder',                   type=str,   help='type of encoder, desenet121_bts, densenet161_bts, '
                                                                    'resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts',
                                                               default='densenet161_bts')
# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
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
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
# Online eval
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--num_threads_eval',          type=int,   default=0)

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

def online_eval(model, normoptizer_eval, crfIntegrater, dataloader_eval, gpu, vlsroot):
    eval_measures_depth = torch.zeros(10).cuda(device=gpu)
    eval_measures_depth_garg = torch.zeros(10).cuda(device=gpu)
    eval_measures_depth_garg_int = torch.zeros(10).cuda(device=gpu)
    eval_measures_shape = torch.zeros(2).cuda(device=gpu)
    for idx, eval_sample_batched in enumerate((dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
            focal = torch.autograd.Variable(eval_sample_batched['focal'].cuda(gpu, non_blocking=True))
            gt_depth = eval_sample_batched['depth'].cuda()
            K = torch.autograd.Variable(eval_sample_batched['K'].cuda(args.gpu, non_blocking=True))

            outputs = model(image, focal)
            pred_depth = outputs['final_depth']
            pred_depth[torch.isinf(pred_depth)] = args.max_depth_eval
            pred_depth[torch.isnan(pred_depth)] = args.min_depth_eval
            int_re, lateral_re, intmask = compute_intre(integrater=crfIntegrater, normoptizer=normoptizer_eval, intrinsic=K, depth_gt=gt_depth,
                                                        shape_pred=outputs['pred_shape'], depth_pred=pred_depth, variance_pred=outputs['pred_variance'], lambda_pred=outputs['pred_lambda'])

            minang = - np.pi / 3 * 2
            maxang = 2 * np.pi - np.pi / 3 * 2

            viewind = 0
            fig_rgb = tensor2rgb(image, viewind=viewind)

            pred_shape = outputs['pred_shape']
            fig_angh = tensor2disp(pred_shape[:, 0].unsqueeze(1) - minang, vmax=maxang, viewind=viewind)
            fig_angv = tensor2disp(pred_shape[:, 1].unsqueeze(1) - minang, vmax=maxang, viewind=viewind)

            pred_depth = outputs['final_depth']
            fig_depth = tensor2disp(1 / pred_depth, vmax=0.15, viewind=viewind)
            fig_intre = tensor2disp(1 / int_re, vmax=0.15, viewind=viewind)

            depth_norm = normoptizer_eval.depth2norm(depthMap=pred_depth, intrinsic=K)
            intre_norm = normoptizer_eval.depth2norm(depthMap=int_re, intrinsic=K)
            fig_depth_norm = tensor2rgb((depth_norm + 1) / 2, isnormed=False)
            fig_intre_norm = tensor2rgb((intre_norm + 1) / 2, isnormed=False)

            fig_variance = tensor2disp(outputs['pred_variance'], vmax=(args.clipvariance + 1), viewind=viewind)
            fig_lambda = tensor2disp(outputs['pred_lambda'], vmax=1, viewind=viewind)

            fignorm = normoptizer_eval.ang2normal(ang=pred_shape, intrinsic=K)
            fignorm = np.array(tensor2rgb((fignorm + 1) / 2, viewind=viewind, isnormed=False))

            figoveiewu = np.concatenate([np.array(fig_rgb), np.array(fignorm)], axis=1)
            figoveiewd = np.concatenate([np.array(fig_angh), np.array(fig_angv)], axis=1)
            figoveiewdd = np.concatenate([np.array(fig_lambda), np.array(fig_variance)], axis=1)
            figoveiewddd = np.concatenate([np.array(fig_depth), np.array(fig_depth_norm)], axis=1)
            figoveiewdddd = np.concatenate([np.array(fig_intre), np.array(fig_intre_norm)], axis=1)
            figoveiew = np.concatenate([figoveiewu, figoveiewd, figoveiewdd, figoveiewddd, figoveiewdddd], axis=0)

            pil.fromarray(figoveiew).save(os.path.join(vlsroot, str(idx).zfill(6) + '.png'))

            pred_depth = pred_depth.cpu().numpy().squeeze()
            pred_depth_int = int_re.cpu().numpy().squeeze()
            pred_shape = outputs['pred_shape']

            # Evaluate for shape
            loss_shape = normoptizer_eval.intergrationloss_ang_validation(ang=pred_shape, intrinsic=K, depthMap=gt_depth)
            gt_depth = gt_depth.cpu().numpy().squeeze()

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth_int[pred_depth_int < args.min_depth_eval] = args.min_depth_eval
        pred_depth_int[pred_depth_int > args.max_depth_eval] = args.max_depth_eval
        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        # Restore Before Kb Crop
        width, height = eval_sample_batched['size'][0].cpu().numpy()
        top_margin = int(height - 352)
        left_margin = int((width - 1216) / 2)
        pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
        pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
        pred_depth_int_uncropped = np.zeros((height, width), dtype=np.float32)
        pred_depth_int_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth_int

        gt_depth_uncropped = np.zeros((height, width), dtype=np.float32)
        gt_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = gt_depth

        valid_mask_kb = np.zeros([height, width])
        valid_mask_kb[int(0.40810811 * height):int(0.99189189 * height), int(0.03594771 * width):int(0.96405229 * width)] = 1
        valid_mask_kb = np.logical_and(valid_mask_kb, np.logical_and(gt_depth_uncropped > args.min_depth_eval, gt_depth_uncropped < args.max_depth_eval))

        # Evaluate on all image
        measures_depth = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
        eval_measures_depth[:9] += torch.tensor(measures_depth).cuda(device=gpu)
        eval_measures_depth[9] += 1

        # Evaluate on garg crop
        measures_depth_garg = compute_errors(gt_depth_uncropped[valid_mask_kb], pred_depth_uncropped[valid_mask_kb])
        eval_measures_depth_garg[:9] += torch.tensor(measures_depth_garg).cuda(device=gpu)
        eval_measures_depth_garg[9] += 1

        # Evaluate on garg crop int results
        measures_depth_garg_int = compute_errors(gt_depth_uncropped[valid_mask_kb], pred_depth_int_uncropped[valid_mask_kb])
        eval_measures_depth_garg_int[:9] += torch.tensor(measures_depth_garg_int).cuda(device=gpu)
        eval_measures_depth_garg_int[9] += 1

        eval_measures_shape[0] += loss_shape
        eval_measures_shape[1] += 1

        print("%d finished" % idx)

    eval_measures_depth[0:9] = eval_measures_depth[0:9] / eval_measures_depth[9]
    eval_measures_depth = eval_measures_depth.cpu().numpy()
    eval_measures_depth_garg[0:9] = eval_measures_depth_garg[0:9] / eval_measures_depth_garg[9]
    eval_measures_depth_garg = eval_measures_depth_garg.cpu().numpy()
    eval_measures_shape = eval_measures_shape[0:1] / eval_measures_shape[1]
    eval_measures_shape = eval_measures_shape.cpu().numpy()
    eval_measures_depth_garg_int[0:9] = eval_measures_depth_garg_int[0:9] / eval_measures_depth_garg_int[9]
    eval_measures_depth_garg_int = eval_measures_depth_garg_int.cpu().numpy()

    print('Computing Depth errors for {} eval samples'.format(int(eval_measures_depth[-1])))
    print("{:>9}, {:>9}, {:>9}, {:>9}, {:>9}, {:>9}, {:>9}, {:>9}, {:>9}".format('silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
    for i in range(8):
        print('{:9.5f}, '.format(eval_measures_depth[i]), end='')
    print('{:9.5f}'.format(eval_measures_depth[8]))

    print('Computing Depth Garg crop errors for {} eval samples'.format(int(eval_measures_depth_garg[-1])))
    print("{:>9}, {:>9}, {:>9}, {:>9}, {:>9}, {:>9}, {:>9}, {:>9}, {:>9}".format('silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
    for i in range(8):
        print('{:9.5f}, '.format(eval_measures_depth_garg[i]), end='')
    print('{:9.5f}'.format(eval_measures_depth_garg[8]))

    print('Computing Depth Garg crop Int errors for {} eval samples'.format(int(eval_measures_depth_garg_int[-1])))
    print("{:>9}, {:>9}, {:>9}, {:>9}, {:>9}, {:>9}, {:>9}, {:>9}, {:>9}".format('silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
    for i in range(8):
        print('{:9.5f}, '.format(eval_measures_depth_garg_int[i]), end='')
    print('{:9.5f}'.format(eval_measures_depth_garg_int[8]))

    print('Computing Shape errors for {} eval samples'.format(int(eval_measures_shape[-1])))
    print('L1 Shape Measurement: %f' % (eval_measures_shape[0]))
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

    # Create model
    model = BtsSDModel(args)
    model = torch.nn.DataParallel(model)
    model.cuda()

    checkpoint = torch.load(os.path.join(args.checkpoint_path))
    model.load_state_dict(checkpoint['model'])

    normoptizer_eval = SurfaceNormalOptimizer(height=kbcroph, width=kbcropw, batch_size=1, angw=args.angw, vlossw=args.vlossw, sclw=args.sclw)
    normoptizer_eval = normoptizer_eval.cuda()
    crfIntegrater = CRFIntegrationModule(clipvariance=args.clipvariance, maxrange=args.maxrange)
    crfIntegrater = crfIntegrater.cuda()

    cudnn.benchmark = True

    dataloader_eval = BtsDataLoader(args, 'online_eval')

    vlsroot = os.path.join('/media/shengjie/disk1/visualization/btspred', args.checkpoint_path.split('/')[-2])
    os.makedirs(vlsroot, exist_ok=True)

    model.eval()
    online_eval(model=model, normoptizer_eval=normoptizer_eval, crfIntegrater=crfIntegrater, dataloader_eval=dataloader_eval, gpu=gpu, vlsroot=vlsroot)

if __name__ == '__main__':
    args = parser.parse_args()
    main_worker(0, 1, args)