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
import sys
import os
import PIL.Image as Image
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__))) # location of src

import torch
import torch.nn as nn
import torch.nn.utils as utils

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import matplotlib
import matplotlib.cm
from tqdm import tqdm
from util import *

from Exp_demon.bts_smoothl1 import BtsModeOrg
from Exp_demon.bts_dataloader import BtsDataLoader
import numpy as np

cudnn.benchmark = True

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        if m.weight.requires_grad == True:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='bts_eigen_v2')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, desenet121_bts, densenet161_bts, '
                                                                    'resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts',
                                                               default='densenet161_bts')
# Dataset
parser.add_argument('--demon_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=416)
parser.add_argument('--input_width',               type=int,   help='input width',  default=544)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=100)

# Log and save
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=500)
parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=2000)

# Random Rotation
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)

# Training
parser.add_argument('--fix_first_conv_blocks',                 help='if set, will fix the first two conv blocks', action='store_true')
parser.add_argument('--fix_first_conv_block',                  help='if set, will fix the first conv block', action='store_true')
parser.add_argument('--bn_no_track_stats',                     help='if set, will not track running stats in batch norm layers', action='store_true')
parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--bts_size',                  type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)

# Multi-gpu training
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                    'N processes per node, which has N GPUs. This is the '
                                                                    'fastest way to use PyTorch for either single node or '
                                                                    'multi node data parallel training', action='store_true',)
# Online eval
parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=0.1)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=100)
parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=2000)
parser.add_argument('--num_threads_eval',          type=int,   default=2)

args = parser.parse_args()
eval_metrics = ['abs_rel', 'abs_diff', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3', 'l1_inv', 'sc_inv']

def compute_errors(gt, pred):
    # same scale
    scale = np.sum(gt) / np.sum(pred)
    pred = pred * scale

    n = float(np.size(gt))
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_diff = np.mean(np.abs(gt - pred))
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    pred = pred * scale
    l1_inv = np.mean(np.abs(np.reciprocal(gt) - np.reciprocal(pred)))

    pred = pred * scale
    log_diff = np.log(gt) - np.log(pred)
    sc_inv = np.sqrt(np.sum(np.square(log_diff)) / n - np.square(np.sum(log_diff)) / np.square(n))
    return [abs_rel, abs_diff, sq_rel, rmse, rmse_log, a1, a2, a3, l1_inv, sc_inv]


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__

def bn_init_as_tf(m):
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = True  # These two lines enable using stats (moving mean and var) loaded from pretrained model
        m.eval()                      # or zero mean and variance of one if the batch norm layer has no pretrained values
        m.affine = True
        m.requires_grad = True

def set_misc(model):
    if args.bn_no_track_stats:
        print("Disabling tracking running stats in batch norm layers")
        model.apply(bn_init_as_tf)

    if args.fix_first_conv_blocks:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', 'base_model.layer1.0', 'base_model.layer1.1', '.bn']
        else:
            fixing_layers = ['conv0', 'denseblock1.denselayer1', 'denseblock1.denselayer2', 'norm']
        print("Fixing first two conv blocks")
    elif args.fix_first_conv_block:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', 'base_model.layer1.0', '.bn']
        else:
            fixing_layers = ['conv0', 'denseblock1.denselayer1', 'norm']
        print("Fixing first conv block")
    else:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', '.bn']
        else:
            fixing_layers = ['conv0', 'norm']
        print("Fixing first conv layer")

    for name, child in model.named_children():
        if not 'encoder' in name:
            continue
        for name2, parameters in child.named_parameters():
            if any(x in name2 for x in fixing_layers):
                parameters.requires_grad = False
                # print("{}-{} frozen".format(name, name2))

def online_eval(model, dataloader_eval, gpu, ngpus):
    eval_measures = {
        'sun3d': torch.zeros(11).cuda(device=gpu),
        'rgbd': torch.zeros(11).cuda(device=gpu),
        'scenes11': torch.zeros(11).cuda(device=gpu),
        'mvs': torch.zeros(11).cuda(device=gpu)
    }
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
            gt_depth = eval_sample_batched['depth']
            _, _, _, _, pred_depth = model(image)
            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        entry = eval_sample_batched['entry'][0][0]
        datasetname = entry.split(' ')[0].split('_')[0]

        gt_depth_ref = np.load(os.path.join('/media/shengjie/disk1/Prediction/EvidentDepthCVPR22/DeMoN/DeepSFMTest_pr', 'gt', "{}.npy".format(entry.replace(' ', '_'))))
        max_diff = np.abs(gt_depth - gt_depth_ref).max()
        assert max_diff < 1e-1

        pred_depth_ref = np.load(os.path.join('/media/shengjie/disk1/Prediction/EvidentDepthCVPR22/DeMoN/DeepSFMTest_pr', 'pr', "{}.npy".format(entry.replace(' ', '_'))))
        pred_depth = pred_depth_ref
        # val1 = gt_depth_ref > 0
        # val2 = gt_depth > 0
        # diffval = np.sum(val1.astype(np.float) - val2.astype(np.float))


        # tensor2disp(1 / torch.from_numpy(gt_depth_ref).unsqueeze(0).unsqueeze(0), vmax=1, viewind=0).show()
        # tensor2disp(1 / torch.from_numpy(gt_depth).unsqueeze(0).unsqueeze(0), vmax=1, viewind=0).show()
        # gt_depth_ref_f = gt_depth_ref[gt_depth_ref > 0]
        # gt_depth_f = gt_depth[gt_depth > 0]
        # compute_errors(gt_depth_ref_f, gt_depth_f)

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[datasetname][:10] += torch.tensor(measures).cuda(device=gpu)
        eval_measures[datasetname][10] += 1

    if args.multiprocessing_distributed:
        group = dist.new_group([i for i in range(ngpus)])
        for k in eval_measures.keys():
            dist.all_reduce(tensor=eval_measures[k], op=dist.ReduceOp.SUM, group=group)

    if not args.multiprocessing_distributed or gpu == 0:
        eval_measures_cpu_dict = dict()
        for k in eval_measures.keys():
            eval_measures_cpu = eval_measures[k].cpu()
            cnt = eval_measures_cpu[10].item()
            eval_measures_cpu /= cnt
            print('Dataset: {}, Computing errors for {} eval samples'.format(k, int(cnt)))
            print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('abs_rel', 'abs_diff', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3', 'l1_inv', 'sc_inv'))
            for i in range(9):
                print('{:7.3f}, '.format(eval_measures_cpu[i]), end='')
            print('{:7.3f}'.format(eval_measures_cpu[9]))
            eval_measures_cpu_dict[k] = eval_measures_cpu
        return eval_measures_cpu_dict
    return None

class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    args.rank = gpu
    args.batch_size = int(args.batch_size / ngpus_per_node)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # Create model
    model = BtsModeOrg(args)
    model.train()
    model.decoder.apply(weights_init_xavier)
    set_misc(model)

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    print("Model Initialized on GPU: {}".format(args.gpu))
    print("Batchsize is %d" % args.batch_size)

    if args.checkpoint_path != '':
        print("Loading checkpoint '{}'".format(args.checkpoint_path))
        loc = 'cuda:{}'.format(args.gpu)
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=loc)['model'])

    dataloader_eval = BtsDataLoader(args, is_test=True)
    eval_measures = online_eval(model, dataloader_eval, gpu, ngpus_per_node)


def main():
    torch.cuda.empty_cache()
    ngpus_per_node = torch.cuda.device_count()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

if __name__ == '__main__':
    main()