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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) # location of src

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch

from tqdm import tqdm
from util import *

from Exp_demon.bts import BtsModeOrg
from Exp_demon.generation.bts_dataloader_generation import BtsDataLoader
import numpy as np

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')

parser.add_argument('--encoder',                   type=str,   help='type of encoder, desenet121_bts, densenet161_bts, '
                                                                    'resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts',
                                                               default='densenet161_bts')
# Dataset
parser.add_argument('--demon_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)

# Log and save
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')

# Training
parser.add_argument('--bts_size',                  type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)

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
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=0.1)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=10)
parser.add_argument('--split',                     type=str, required=True)
parser.add_argument('--export_root',               type=str, required=True)

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

def online_eval(model, dataloader_eval, gpu, ngpus, entrynum):
    eval_measures = torch.zeros(11).cuda(device=gpu)
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
            gt_depth = eval_sample_batched['depth']
            _, _, _, _, pred_depth = model(image)
            pred_depth = pred_depth.cpu().numpy().squeeze(1)
            gt_depth = gt_depth.cpu().numpy().squeeze(1)

            pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
            pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
            pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
            pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        for k in range(pred_depth.shape[0]):
            datasetentry = eval_sample_batched['entry'][0][k].split(' ')[0].split('_')[0]
            if datasetentry == 'scenes11':
                valid_mask = np.logical_and(gt_depth[k] > args.min_depth_eval, gt_depth[k] < 50)
            elif datasetentry == 'mvs':
                valid_mask = np.logical_and(gt_depth[k] > args.min_depth_eval, gt_depth[k] < 100)
            else:
                valid_mask = np.logical_and(gt_depth[k] > args.min_depth_eval, gt_depth[k] < 10)

            measures = compute_errors(gt_depth[k][valid_mask], pred_depth[k][valid_mask])

            eval_measures[:10] += torch.tensor(measures).cuda(device=gpu)
            eval_measures[10] += 1

    if args.multiprocessing_distributed:
        group = dist.new_group([i for i in range(ngpus)])
        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

    if not args.multiprocessing_distributed or gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[10].item()
        eval_measures_cpu[0:10] /= cnt

        assert entrynum == cnt
    return eval_measures_cpu

def online_generation(model, dataloader_eval, gpu, ss):
    scale_list = list()
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
            gt_depth = eval_sample_batched['depth']
            _, _, _, _, pred_depth = model(image)

            pred_depth = pred_depth.cpu().numpy().squeeze(1)
            gt_depth = gt_depth.cpu().numpy().squeeze(1)

            pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
            pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        for k in range(pred_depth.shape[0]):
            datasetentry = eval_sample_batched['entry'][0][k].split(' ')[0].split('_')[0]
            if datasetentry == 'scenes11':
                valid_mask = np.logical_and(gt_depth[k] > args.min_depth_eval, gt_depth[k] < 50)
            elif datasetentry == 'mvs':
                valid_mask = np.logical_and(gt_depth[k] > args.min_depth_eval, gt_depth[k] < 100)
            else:
                valid_mask = np.logical_and(gt_depth[k] > args.min_depth_eval, gt_depth[k] < 10)

            assert np.sum(valid_mask) > 10, print("Invalid Groundtruth")

            scale = np.sum(gt_depth[k][valid_mask]) / np.sum(pred_depth[k][valid_mask])

            pred_depth_sv = pred_depth[k] * scale
            pred_depth_sv = np.clip(pred_depth_sv, a_min=0.1, a_max=100)

            svfold = os.path.join(args.export_root, ss, eval_sample_batched['entry'][0][k].split(' ')[0])
            os.makedirs(svfold, exist_ok=True)

            pred_depth_sv = pred_depth_sv * 256.0
            pred_depth_sv = pred_depth_sv.astype(np.uint16)
            svpath = os.path.join(svfold, "{}.png".format(eval_sample_batched['entry'][0][k].split(' ')[1].zfill(4)))
            Image.fromarray(pred_depth_sv).save(svpath)

            # pred_depth_sv_ck = np.array(Image.open(svpath)).astype(np.float32)
            # pred_depth_sv_ck = pred_depth_sv_ck / 256.0

            scale_list.append(scale)
    return scale_list

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
    model.eval()

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    print("Model Initialized on GPU: {}".format(args.gpu))

    args.checkpoint_path = os.path.join(args.checkpoint_path, "model_scinv_{}.pth".format(args.split))
    if args.checkpoint_path != '':
        print("Loading checkpoint '{}'".format(args.checkpoint_path))
        loc = 'cuda:{}'.format(args.gpu)
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=loc)['model'])

    dataloader_eval = BtsDataLoader(args, args.split, batch_size=8, num_workers=0, jitterparam=0, is_test=True, verbose=True, islimit=True)
    eval_measures_cpu = online_eval(model, dataloader_eval, gpu, ngpus_per_node, entrynum=dataloader_eval.demon.__len__())

    if gpu == 0:
        print("Split: %s, Counts: %d, abs_rel: %f, abs_diff: %f, a1: %f, l1_inv: %f, sc_inv: %f" % (args.split, eval_measures_cpu[10], eval_measures_cpu[0], eval_measures_cpu[1], eval_measures_cpu[5], eval_measures_cpu[8], eval_measures_cpu[9]))

    train_scinv_aug = list()
    train_measures_cpu_rec = list()
    jitterparams = np.linspace(0, 1.2, 12)

    if args.split in ['sun3d', 'rgbd']:
        for jitterparam in jitterparams:
            dataloader_train = BtsDataLoader(args, args.split, batch_size=8, num_workers=2, jitterparam=jitterparam, is_test=False, verbose=False, islimit=True)
            train_measures_cpu = online_eval(model, dataloader_train, gpu, ngpus_per_node, entrynum=dataloader_train.demon.__len__())
            if gpu == 0:
                print("Split: %s, Aug: %f, Counts: %d, abs_rel: %f, abs_diff: %f, a1: %f, l1_inv: %f, sc_inv: %f" % (args.split, jitterparam, train_measures_cpu[10], train_measures_cpu[0], train_measures_cpu[1], train_measures_cpu[5], train_measures_cpu[8], train_measures_cpu[9]))
            train_scinv_aug.append(train_measures_cpu[9])
            train_measures_cpu_rec.append(train_measures_cpu)

        train_scinv_aug = np.array(train_scinv_aug)
        opt_jitterparam = np.argmin(np.abs(train_scinv_aug - eval_measures_cpu[9]))

        if gpu == 0:
            train_measures_cpu = train_measures_cpu_rec[opt_jitterparam]
            print("====================================================================================================")
            print("Split: %s, Counts: %d, abs_rel: %f, abs_diff: %f, a1: %f, l1_inv: %f, sc_inv: %f" % (args.split, eval_measures_cpu[10], eval_measures_cpu[0], eval_measures_cpu[1], eval_measures_cpu[5], eval_measures_cpu[8], eval_measures_cpu[9]))
            print("Split: %s, Aug: %f, Counts: %d, abs_rel: %f, abs_diff: %f, a1: %f, l1_inv: %f, sc_inv: %f" % (args.split, jitterparams[opt_jitterparam], train_measures_cpu[10], train_measures_cpu[0], train_measures_cpu[1], train_measures_cpu[5], train_measures_cpu[8], train_measures_cpu[9]))
    else:
        opt_jitterparam = 0

    dataloader_generation_test = BtsDataLoader(args, args.split, batch_size=8, num_workers=4, jitterparam=0, is_test=True, verbose=True, islimit=False)
    online_generation(model, dataloader_generation_test, gpu, ss='test')

    dataloader_generation_train = BtsDataLoader(args, args.split, batch_size=8, num_workers=4, jitterparam=jitterparams[opt_jitterparam], is_test=False, verbose=True, islimit=False)
    scale_list = online_generation(model, dataloader_generation_train, gpu, ss='train')
    scale_list = np.array(scale_list)

    plt.figure()
    plt.hist(scale_list, bins=20)
    plt.savefig(os.path.join(args.export_root, "{}.png".format(args.split)))
    plt.close()


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