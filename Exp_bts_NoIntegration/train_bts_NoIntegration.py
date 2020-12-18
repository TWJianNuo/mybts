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

from Exp_bts_NoIntegration.btsnet import BtsSDModel
from Exp_bts_NoIntegration.bts_dataloader import BtsDataLoader
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

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='bts_eigen_v2')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, desenet121_bts, densenet161_bts, '
                                                                    'resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts',
                                                               default='densenet161_bts')
# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=True)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)

# Log and save
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=500)

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
parser.add_argument('--learning_rate_shape',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)

# Preprocessing
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

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
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=500)
parser.add_argument('--num_threads_eval',          type=int,   default=2)

# Integration
parser.add_argument("--inttimes",                  type=int,     default=1)
parser.add_argument("--clipvariance",              type=float,   default=5)
parser.add_argument("--maxrange",                  type=float,   default=100)

# Shape
parser.add_argument('--angw',                      type=float, default=0)
parser.add_argument('--vlossw',                    type=float, default=1)
parser.add_argument('--sclw',                      type=float, default=1e-3)
parser.add_argument("--scheduler_step_size",       type=int,   help="step size of the scheduler", default=15)

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.mode == 'train' and not args.checkpoint_path:
    from bts import *

elif args.mode == 'train' and args.checkpoint_path:
    model_dir = os.path.dirname(args.checkpoint_path)
    model_name = os.path.basename(model_dir)
    import sys
    sys.path.append(model_dir)
    for key, val in vars(__import__(model_name)).items():
        if key.startswith('__') and key.endswith('__'):
            continue
        vars()[key] = val

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

def colorize(value, vmin=None, vmax=None, cmap='Greys'):
    value = value.cpu().numpy()[:, :, :]
    value = np.log10(value)

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value*0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))


def normalize_result(value, vmin=None, vmax=None):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    return np.expand_dims(value, 0)


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
        if not 'depth_encoder' in name:
            continue
        for name2, parameters in child.named_parameters():
            if any(x in name2 for x in fixing_layers):
                parameters.requires_grad = False
                print("{}-{} frozen".format(name, name2))

def online_eval(model, normoptizer_eval, crfIntegrater, dataloader_eval, gpu, ngpus):
    eval_measures_depth = torch.zeros(10).cuda(device=gpu)
    eval_measures_depth_garg = torch.zeros(10).cuda(device=gpu)
    eval_measures_shape = torch.zeros(2).cuda(device=gpu)
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
            focal = torch.autograd.Variable(eval_sample_batched['focal'].cuda(gpu, non_blocking=True))
            gt_depth = eval_sample_batched['depth'].cuda()
            K = torch.autograd.Variable(eval_sample_batched['K'].cuda(args.gpu, non_blocking=True))

            outputs = model(image, focal)
            pred_depth = outputs['final_depth'].cpu().numpy().squeeze()
            pred_shape = outputs['pred_shape']

            # Evaluate for shape
            loss_shape = normoptizer_eval.intergrationloss_ang_validation(ang=pred_shape, intrinsic=K, depthMap=gt_depth)
            gt_depth = gt_depth.cpu().numpy().squeeze()

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval
        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        # Restore Before Kb Crop
        width, height = eval_sample_batched['size'][0].cpu().numpy()
        top_margin = int(height - 352)
        left_margin = int((width - 1216) / 2)
        pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
        pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth

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

        eval_measures_shape[0] += loss_shape
        eval_measures_shape[1] += 1

    if args.multiprocessing_distributed:
        group = dist.new_group([i for i in range(ngpus)])
        dist.all_reduce(tensor=eval_measures_depth, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=eval_measures_depth_garg, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(tensor=eval_measures_shape, op=dist.ReduceOp.SUM, group=group)

    if not args.multiprocessing_distributed or gpu == 0:
        eval_measures_depth[0:9] = eval_measures_depth[0:9] / eval_measures_depth[9]
        eval_measures_depth = eval_measures_depth.cpu().numpy()
        eval_measures_depth_garg[0:9] = eval_measures_depth_garg[0:9] / eval_measures_depth_garg[9]
        eval_measures_depth_garg = eval_measures_depth_garg.cpu().numpy()
        eval_measures_shape = eval_measures_shape[0:1] / eval_measures_shape[1]
        eval_measures_shape = eval_measures_shape.cpu().numpy()

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

        print('Computing Shape errors for {} eval samples'.format(int(eval_measures_shape[-1])))
        print('L1 Shape Measurement: %f' % (eval_measures_shape[0]))
        return eval_measures_depth, eval_measures_depth_garg, eval_measures_shape
    return None

def compute_shape_loss(normoptizer, pred_shape, intrinsic, depth_gt):
    loss, _, _, _, _ = normoptizer.intergrationloss_ang(ang=pred_shape, intrinsic=intrinsic, depthMap=depth_gt)
    return loss

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # Create model
    model = BtsSDModel(args)
    model.train()
    model.depth_decoder.apply(weights_init_xavier)
    set_misc(model)

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("Total number of learning parameters: {}".format(num_params_update))

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

            normoptizer = SurfaceNormalOptimizer(height=args.input_height, width=args.input_width, batch_size=int(args.batch_size / dist.get_world_size()), angw=args.angw, vlossw=args.vlossw, sclw=args.sclw)
            normoptizer_eval = SurfaceNormalOptimizer(height=kbcroph, width=kbcropw, batch_size=1, angw=args.angw, vlossw=args.vlossw, sclw=args.sclw)
            normoptizer.to(f'cuda:{args.gpu}')
            normoptizer_eval.to(f'cuda:{args.gpu}')
            crfIntegrater = CRFIntegrationModule(clipvariance=args.clipvariance, maxrange=args.maxrange)
            crfIntegrater.to(f'cuda:{args.gpu}')
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

            normoptizer = SurfaceNormalOptimizer(height=args.input_height, width=args.input_width, batch_size=int(args.batch_size), angw=args.angw, vlossw=args.vlossw, sclw=args.sclw)
            normoptizer_eval = SurfaceNormalOptimizer(height=kbcroph, width=kbcropw, batch_size=1, angw=args.angw, vlossw=args.vlossw, sclw=args.sclw)
            normoptizer = normoptizer.cuda()
            normoptizer_eval = normoptizer_eval.cuda()
            crfIntegrater = CRFIntegrationModule(clipvariance=args.clipvariance, maxrange=args.maxrange)
            crfIntegrater = crfIntegrater.cuda()
    else:
        model = torch.nn.DataParallel(model)
        model.cuda()

        normoptizer = SurfaceNormalOptimizer(height=args.input_height, width=args.input_width, batch_size=int(args.batch_size), angw=args.angw, vlossw=args.vlossw, sclw=args.sclw)
        normoptizer_eval = SurfaceNormalOptimizer(height=kbcroph, width=kbcropw, batch_size=1, angw=args.angw, vlossw=args.vlossw, sclw=args.sclw)
        normoptizer = normoptizer.cuda()
        normoptizer_eval = normoptizer_eval.cuda()
        crfIntegrater = CRFIntegrationModule(clipvariance=args.clipvariance, maxrange=args.maxrange)
        crfIntegrater = crfIntegrater.cuda()

    if args.distributed:
        print("Model Initialized on GPU: {}".format(args.gpu))
    else:
        print("Model Initialized")

    global_step = 0
    best_measures = np.zeros([3])
    best_steps = np.zeros([3])
    best_measures[0] = 1e3
    best_measures[1] = 1e3
    best_measures[2] = 0

    # Training parameters
    optimizer = torch.optim.AdamW([{'params': model.module.depth_encoder.parameters(), 'weight_decay': args.weight_decay, 'lr': args.learning_rate, 'eps': args.adam_eps, 'name': 'depth_encoder'},
                                   {'params': model.module.depth_decoder.parameters(), 'weight_decay': 0, 'lr': args.learning_rate, 'eps': args.adam_eps, 'name': 'depth_decoder'},
                                   {'params': list(model.module.shape_encoder.parameters()) + list(model.module.shape_decoder.parameters()), 'weight_decay': 0, 'lr': args.learning_rate_shape, 'eps': args.adam_eps, 'name': 'shapenet'}])

    model_just_loaded = False
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("Loading checkpoint '{}'".format(args.checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(args.checkpoint_path)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint_path, map_location=loc)
            global_step = checkpoint['global_step']
            model.load_state_dict(checkpoint['model'])
            print("Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path, checkpoint['global_step']))
        else:
            print("No checkpoint found at '{}'".format(args.checkpoint_path))
        model_just_loaded = True

    if args.retrain:
        global_step = 0

    cudnn.benchmark = True

    dataloader = BtsDataLoader(args, 'train')
    dataloader_eval = BtsDataLoader(args, 'online_eval')

    # Logging
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(os.path.join(args.log_directory, args.model_name, 'summaries'), flush_secs=30)
        eval_summary_writer_Depth = SummaryWriter(os.path.join(args.log_directory, 'eval_Depth', args.model_name), flush_secs=30)
        eval_summary_writer_Depth_garg = SummaryWriter(os.path.join(args.log_directory, 'eval_Depth', "{}_{}".format(args.model_name, 'garg')), flush_secs=30)
        eval_summary_writer_Shape = SummaryWriter(os.path.join(args.log_directory, 'eval', args.model_name), flush_secs=30)

    silog_criterion = silog_loss(variance_focus=args.variance_focus)

    start_time = time.time()
    duration = 0

    var_sum = [var.sum() for var in model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)

    print("Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum/var_cnt))

    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    num_lrmod_steps = (args.num_epochs - 5) * steps_per_epoch
    epoch = global_step // steps_per_epoch

    measurements = ['ShapeL1', 'DepthAbsrelGarg', 'DepthA1Garg']

    while epoch < args.num_epochs:
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)

        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()

            before_op_time = time.time()

            image = torch.autograd.Variable(sample_batched['image'].cuda(args.gpu, non_blocking=True))
            focal = torch.autograd.Variable(sample_batched['focal'].cuda(args.gpu, non_blocking=True))
            depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(args.gpu, non_blocking=True))
            K = torch.autograd.Variable(sample_batched['K'].cuda(args.gpu, non_blocking=True))

            outputs = model(image, focal)

            mask = depth_gt > 1.0

            loss_depth = silog_criterion.forward(outputs['final_depth'], depth_gt, mask.to(torch.bool))
            loss_shape = compute_shape_loss(normoptizer, outputs['pred_shape'], K, depth_gt)
            loss = loss_depth + loss_shape
            loss.backward()

            for param_group in optimizer.param_groups:
                if param_group['name'] != 'shapenet':
                    current_lr = (args.learning_rate - 0.1 * args.learning_rate) * (1 - global_step / num_lrmod_steps) ** 0.9 + 0.1 * args.learning_rate
                else:
                    current_lr = (args.learning_rate_shape - 0.1 * args.learning_rate_shape) * (1 - global_step / num_lrmod_steps) ** 0.9 + 0.1 * args.learning_rate_shape
                param_group['lr'] = current_lr

            optimizer.step()

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}'.format(epoch, step, steps_per_epoch, global_step, current_lr, loss))
                if np.isnan(loss.cpu().item()):
                    print('NaN in loss occurred. Aborting training.')
                    return -1

            duration += time.time() - before_op_time
            if global_step and global_step % args.log_freq == 0 and not model_just_loaded:
                var_sum = [var.sum() for var in model.parameters() if var.requires_grad]
                var_cnt = len(var_sum)
                var_sum = np.sum(var_sum)
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    print("{}".format(args.model_name))
                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | var sum: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(args.gpu, examples_per_sec, loss, var_sum.item(), var_sum.item()/var_cnt, time_sofar, training_time_left))

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    minang = - np.pi / 3 * 2
                    maxang = 2 * np.pi - np.pi / 3 * 2

                    viewind = 0
                    depth_gtvls = torch.clone(depth_gt)
                    depth_gtvls[depth_gtvls == 0] = float("Inf")
                    fig_gt = tensor2disp_circ(1 / depth_gtvls, vmax=0.15, viewind=viewind)
                    fig_rgb = tensor2rgb(sample_batched['image'], viewind=viewind)

                    pred_shape = outputs['pred_shape']
                    fig_angh = tensor2disp(pred_shape[:, 0].unsqueeze(1) - minang, vmax=maxang, viewind=viewind)
                    fig_angv = tensor2disp(pred_shape[:, 1].unsqueeze(1) - minang, vmax=maxang, viewind=viewind)

                    pred_depth = outputs['final_depth']
                    fig_depth = tensor2disp(1/pred_depth, vmax=0.15, viewind=viewind)

                    fig_variance = tensor2disp(outputs['pred_variance'], vmax=(args.clipvariance + 1), viewind=viewind)
                    fig_lambda = tensor2disp(outputs['pred_lambda'], vmax=1, viewind=viewind)

                    fignorm = normoptizer.ang2normal(ang=pred_shape, intrinsic=K)
                    fignorm = np.array(tensor2rgb((fignorm + 1) / 2, viewind=viewind, isnormed=False))

                    figoveiewu = np.concatenate([np.array(fig_rgb), np.array(fig_gt)], axis=1)
                    figoveiewd = np.concatenate([np.array(fig_angh), np.array(fig_angv)], axis=1)
                    figoveiewdd = np.concatenate([np.array(fig_depth), np.array(fignorm)], axis=1)
                    figoveiewddd = np.concatenate([np.array(fig_lambda), np.array(fig_variance)], axis=1)
                    figoveiew = np.concatenate([figoveiewu, figoveiewd, figoveiewdd, figoveiewddd], axis=0)

                    writer.add_image('oview', (torch.from_numpy(figoveiew).float() / 255).permute([2, 0, 1]), global_step)
                    if version_num > 1100000000:
                        writer.flush()

            if not args.do_online_eval and global_step and global_step % args.save_freq == 0:
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    checkpoint = {'global_step': global_step,
                                  'model': model.state_dict()}
                    torch.save(checkpoint, args.log_directory + '/' + args.model_name + '/model-{}'.format(global_step))

            if args.do_online_eval and global_step and global_step % args.eval_freq == 0 and not model_just_loaded:
                time.sleep(0.1)
                model.eval()
                eval_measures_depth, eval_measures_depth_garg, eval_measures_shape = online_eval(model=model, normoptizer_eval=normoptizer_eval, crfIntegrater=crfIntegrater, dataloader_eval=dataloader_eval, gpu=gpu, ngpus=ngpus_per_node)
                eval_summary_writer_Shape.add_scalar('L1Measure_semidense', eval_measures_shape[0], int(global_step))
                eval_summary_writer_Depth.add_scalar('Depth_absrel_semidense', eval_measures_depth[1], int(global_step))
                eval_summary_writer_Depth.add_scalar('Depth_a1_semidense', eval_measures_depth[6], int(global_step))
                eval_summary_writer_Depth_garg.add_scalar('Depth_absrel_semidense_garg', eval_measures_depth_garg[1], int(global_step))
                eval_summary_writer_Depth_garg.add_scalar('Depth_a1_semidense_garg', eval_measures_depth_garg[6], int(global_step))

                if epoch >= 10:
                    for kk in range(best_measures.shape[0]):
                        is_best = False
                        if kk == 0 and eval_measures_shape[0] < best_measures[kk]:
                            old_best_measure = best_measures[kk]
                            old_best_step = best_steps[kk]
                            best_measures[kk] = eval_measures_shape[0]
                            best_steps[kk] = global_step
                            is_best = True
                        elif kk == 1 and eval_measures_depth_garg[1] < best_measures[kk]:
                            old_best_measure = best_measures[kk]
                            old_best_step = best_steps[kk]
                            best_measures[kk] = eval_measures_depth_garg[1]
                            best_steps[kk] = global_step
                            is_best = True
                        elif kk == 2 and eval_measures_depth_garg[6] > best_measures[kk]:
                            old_best_measure = best_measures[kk]
                            old_best_step = best_steps[kk]
                            best_measures[kk] = eval_measures_depth_garg[6]
                            best_steps[kk] = global_step
                            is_best = True
                        if is_best:
                            old_best_name = 'model-{}-best_{}_{:.5f}'.format(old_best_step, measurements[kk], old_best_measure)
                            model_path = os.path.join(args.log_directory, args.model_name, old_best_name)
                            if os.path.exists(model_path):
                                command = 'rm {}'.format(model_path)
                                os.system(command)
                            model_save_name = 'model-{}-best_{}_{:.5f}'.format(best_steps[kk], measurements[kk], best_measures[kk])
                            print('New best for {}. Saving model: {}'.format(measurements[kk], model_save_name))
                            checkpoint = {'global_step': global_step,
                                          'model': model.state_dict(),
                                          'best_measures': best_measures,
                                          'best_steps': best_steps
                                          }
                            torch.save(checkpoint, os.path.join(args.log_directory, args.model_name, model_save_name))
                        if version_num > 1100000000:
                            eval_summary_writer_Shape.flush()
                            eval_summary_writer_Depth.flush()
                            eval_summary_writer_Depth_garg.flush()
                print("Best Shape L1: %f, at step %d" % (best_measures[0], best_steps[0]))
                print("Best Depth abserl: %f, at step %d" % (best_measures[1], best_steps[1]))
                print("Best Depth a1: %f, at step %d" % (best_measures[2], best_steps[2]))

                model.train()
                block_print()
                set_misc(model)
                enable_print()

            model_just_loaded = False
            global_step += 1

        epoch += 1

def main():
    if args.mode != 'train':
        print('bts_main.py is only for training. Use bts_test.py instead.')
        return -1

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print("This will evaluate the model every eval_freq {} steps and save best models for individual eval metrics." .format(args.eval_freq))

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()