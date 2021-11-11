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

from Exp_demon.bts import BtsModeOrg
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
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)

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
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=0.5)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=50)
parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=500)
parser.add_argument('--num_threads_eval',          type=int,   default=2)

args = parser.parse_args()
eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']

# def compute_errors(gt, pred):
#     thresh = np.maximum((gt / pred), (pred / gt))
#     d1 = (thresh < 1.25).mean()
#     d2 = (thresh < 1.25 ** 2).mean()
#     d3 = (thresh < 1.25 ** 3).mean()
#
#     rms = (gt - pred) ** 2
#     rms = np.sqrt(rms.mean())
#
#     log_rms = (np.log(gt) - np.log(pred)) ** 2
#     log_rms = np.sqrt(log_rms.mean())
#
#     abs_rel = np.mean(np.abs(gt - pred) / gt)
#     sq_rel = np.mean(((gt - pred) ** 2) / gt)
#
#     err = np.log(pred) - np.log(gt)
#     silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
#
#     err = np.abs(np.log10(pred) - np.log10(gt))
#     log10 = np.mean(err)
#
#     return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]

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
        'mvs': torch.zeros(11).cuda(device=gpu),
        'rgbd': torch.zeros(11).cuda(device=gpu),
        'scenes11': torch.zeros(11).cuda(device=gpu)
    }
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
            gt_depth = eval_sample_batched['depth']
            _, _, _, _, pred_depth = model(image)
            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        entry = eval_sample_batched['entry'][0][0]
        datasetname = entry.split(' ')[0].split('_')[0]

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

    global_step = 0
    best_eval_measures_lower_better = torch.zeros(6).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(9, dtype=np.int32)
    # best_eval_measures = dict()

    # Training parameters
    optimizer = torch.optim.AdamW([{'params': model.module.encoder.parameters(), 'weight_decay': args.weight_decay},
                                   {'params': model.module.decoder.parameters(), 'weight_decay': 0}], lr=args.learning_rate, eps=args.adam_eps)

    if args.checkpoint_path != '':
        print("Loading checkpoint '{}'".format(args.checkpoint_path))
        loc = 'cuda:{}'.format(args.gpu)
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=loc)['model'])

    dataloader = BtsDataLoader(args, is_test=False)
    dataloader_eval = BtsDataLoader(args, is_test=True)

    # Logging
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(os.path.join(args.log_directory, args.model_name, 'summaries'), flush_secs=30)

    silog_criterion = silog_loss(variance_focus=args.variance_focus)

    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate

    defaul_max_epoch = 50

    epoch = 0
    steps_per_epoch = np.ceil(dataloader.demon.__len__() / args.batch_size).astype(np.int32)
    num_total_steps = defaul_max_epoch * steps_per_epoch
    while epoch < args.num_epochs:

        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)

        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()

            image = torch.autograd.Variable(sample_batched['image'].cuda(args.gpu, non_blocking=True))
            depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(args.gpu, non_blocking=True))

            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image)

            mask = (depth_gt > 0) * (depth_gt < args.max_depth)

            loss = silog_criterion.forward(depth_est, depth_gt, mask.to(torch.bool))
            loss.backward()
            for param_group in optimizer.param_groups:
                current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                if current_lr < end_learning_rate:
                    current_lr = end_learning_rate
                param_group['lr'] = current_lr

            optimizer.step()

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0 and global_step % args.log_freq == 0):
                print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}'.format(epoch, step, steps_per_epoch, global_step, current_lr, loss))
                if np.isnan(loss.cpu().item()):
                    print('NaN in loss occurred. Aborting training.')
                    return -1

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                writer.add_scalar('Train/silog_loss', loss, global_step)
                writer.add_scalar('Train/learning_rate', current_lr, global_step)

            if global_step % args.log_freq == 0:
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    viewind = 0
                    depth_gt_ = torch.clone(depth_gt)
                    depth_gt_[depth_gt_ == 0] = float("Inf")

                    outrange_sel = mask[viewind].squeeze().cpu().numpy() == 0

                    fig_gt = tensor2disp(1 / depth_gt_, vmax=1, viewind=viewind)
                    fig_rgb = tensor2rgb(sample_batched['image'], viewind=viewind)
                    fig_depth = tensor2disp(1 / depth_est, vmax=1, viewind=viewind)

                    fig_rgb = np.array(fig_rgb)
                    fig_rgb[outrange_sel, 0] = fig_rgb[outrange_sel, 0] * 0.5 + 255.0 * 0.5

                    figoveiew = np.concatenate([fig_rgb, fig_gt, fig_depth], axis=0)

                    writer.add_image('oview', (torch.from_numpy(figoveiew).float() / 255).permute([2, 0, 1]), global_step)

            if not args.do_online_eval and global_step and global_step % args.save_freq == 0:
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    checkpoint = {'model': model.state_dict()}
                    torch.save(checkpoint, args.log_directory + '/' + args.model_name + '/model-{}'.format(global_step))

            if args.do_online_eval and global_step % args.eval_freq == 0:
                time.sleep(0.1)
                model.eval()
                eval_measures = online_eval(model, dataloader_eval, gpu, ngpus_per_node)
                if eval_measures is not None:
                    for i in range(9):
                        writer.add_scalar("Test/{}".format(eval_metrics[i]), eval_measures[i].cpu(), int(global_step))
                        measure = eval_measures[i]
                        is_best = False
                        if i < 6 and measure < best_eval_measures_lower_better[i]:
                            old_best = best_eval_measures_lower_better[i].item()
                            best_eval_measures_lower_better[i] = measure.item()
                            is_best = True
                        elif i >= 6 and measure > best_eval_measures_higher_better[i-6]:
                            old_best = best_eval_measures_higher_better[i-6].item()
                            best_eval_measures_higher_better[i-6] = measure.item()
                            is_best = True
                        if is_best:
                            old_best_step = best_eval_steps[i]
                            old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, eval_metrics[i], old_best)
                            model_path = args.log_directory + '/' + args.model_name + old_best_name
                            if os.path.exists(model_path):
                                command = 'rm {}'.format(model_path)
                                os.system(command)
                            best_eval_steps[i] = global_step
                            model_save_name = '/model-{}-best_{}_{:.5f}'.format(global_step, eval_metrics[i], measure)
                            print('New best for {}. Saving model: {}'.format(eval_metrics[i], model_save_name))
                            checkpoint = {'global_step': global_step,
                                          'model': model.state_dict(),
                                          'optimizer': optimizer.state_dict(),
                                          'best_eval_measures_higher_better': best_eval_measures_higher_better,
                                          'best_eval_measures_lower_better': best_eval_measures_lower_better,
                                          'best_eval_steps': best_eval_steps
                                          }
                            torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)

                    print('Best Performance:')
                    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
                    for i in range(9):
                        if i < 6:
                            print('{:7.3f}, '.format(best_eval_measures_lower_better[i]), end='')
                        elif i != 8:
                            print('{:7.3f}'.format(best_eval_measures_higher_better[i-6]), end='')
                        else:
                            print('{:7.3f}'.format(best_eval_measures_higher_better[i - 6]))

                model.train()
                block_print()
                set_misc(model)
                enable_print()

            global_step += 1

        epoch += 1


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