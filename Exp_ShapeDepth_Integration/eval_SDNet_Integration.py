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

import os

import time
import argparse
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__))) # location of src

import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

import matplotlib
import matplotlib.cm
from tqdm import tqdm

from Exp_ShapeDepth_Integration.shapedataset import KittiShapeDataLoader, KittiShapeDataset
from Exp_ShapeDepth_Integration.SDNet import ShapeNet
from util import *
from torchvision import transforms
import torch.utils.data.distributed
import torch.distributed as dist
from integrationModule import CRFIntegrationModule

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='bts_eigen_v2')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, desenet121_bts, densenet161_bts, resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts', default='densenet161_bts')
# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=True)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)

# Log and save
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=500)

# Training
parser.add_argument('--fix_first_conv_blocks',                 help='if set, will fix the first two conv blocks', action='store_true')
parser.add_argument('--fix_first_conv_block',                  help='if set, will fix the first conv block', action='store_true')
parser.add_argument('--bn_no_track_stats',                     help='if set, will not track running stats in batch norm layers', action='store_true')
parser.add_argument('--bts_size',                  type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument("--scheduler_step_size",       type=int,   help="step size of the scheduler", default=15)
parser.add_argument('--angw',                      type=float, default=0)
parser.add_argument('--vlossw',                    type=float, default=1)
parser.add_argument('--sclw',                      type=float, default=1e-3)
parser.add_argument('--min_depth',                 type=float, help="min depth value", default=0.1)
parser.add_argument('--max_depth',                 type=float, help="max depth value", default=100)
parser.add_argument('--depthlossw',                type=float, help="weight of loss on depth", default=1e-2)
parser.add_argument('--variancelossw',             type=float, help="mounted to depth loss", default=1)
parser.add_argument('--startstep',                 type=int,   help="mounted to depth loss", default=5000)

parser.add_argument("--inttimes",               type=int,     default=1)
parser.add_argument("--clipvariance",           type=float,   default=5)
parser.add_argument("--maxrange",               type=float,   default=100)

# Preprocessing
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Multi-gpu training
parser.add_argument('--num_workers',               type=int,   help='number of threads to use for data loading', default=1)
parser.add_argument('--num_workers_eval',          type=int,   help='number of threads to use for eval data loading', default=2)
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
parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=500)
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)

kbcroph = 352
kbcropw = 1216

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

def online_eval(model, normoptizer_eval, crfIntegrater, dataloader_eval, gpu, ngpus):
    eval_measures_depth = torch.zeros(10)
    eval_measures_lateralre = torch.zeros(10)
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            if 'depth' not in eval_sample_batched:
                continue
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(args.gpu, non_blocking=True))
            K = torch.autograd.Variable(eval_sample_batched['K'].cuda(gpu, non_blocking=True))
            depth_gt = eval_sample_batched['depth'].cuda()
            outputs = model(image)

            tensor2disp(outputs[('variance', 0)], vmax=args.clipvariance).show()
            tensor2disp(outputs[('lambda', 0)], vmax=1).show()

            # Compute Measurement for Lateralre
            compute_intre(integrater=crfIntegrater, normoptizer=normoptizer_eval, outputs=outputs, intrinsic=K, depth_gt=depth_gt, scales=1)

            depth_gt = depth_gt.cpu()

            valid_mask = (depth_gt > args.min_depth_eval) * (depth_gt < args.max_depth_eval)
            valid_mask = valid_mask == 1
            valid_mask = valid_mask.cpu()

            depth_latrealre = outputs[('intre', 0)]
            depth_latrealre = depth_latrealre.cpu()
            depth_latrealre = torch.clamp(depth_latrealre, min=args.min_depth_eval, max=args.max_depth_eval)
            depth_latrealre_flatten = depth_latrealre[valid_mask].numpy()


            # Compute Measurement for Depth
            pred_depth = outputs[('depth', 0)]
            pred_depth = torch.clamp(pred_depth, min=args.min_depth_eval, max=args.max_depth_eval)
            pred_depth = pred_depth.cpu()
            pred_depth_flatten = pred_depth[valid_mask].numpy()
            depth_gt_flatten = depth_gt[valid_mask].numpy()

            eval_measures_depth_np = compute_errors(gt=depth_gt_flatten, pred=pred_depth_flatten)
            eval_measures_depth[:9] += torch.tensor(eval_measures_depth_np)

            eval_measures_lateralre_np = compute_errors(gt=depth_gt_flatten, pred=depth_latrealre_flatten)
            eval_measures_lateralre[:9] += torch.tensor(eval_measures_lateralre_np)

            eval_measures_depth[9] += 1
            eval_measures_lateralre[9] += 1

    eval_measures_depth[0:9] = eval_measures_depth[0:9] / eval_measures_depth[9]
    eval_measures_lateralre[0:9] = eval_measures_lateralre[0:9] / eval_measures_lateralre[9]
    eval_measures_depth = eval_measures_depth.cpu().numpy()
    eval_measures_lateralre = eval_measures_lateralre.cpu().numpy()
    print('Computing Depth errors for {} eval samples'.format(int(eval_measures_depth[-1])))
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>9}, {:>9}, {:>9}".format('silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
    for i in range(8):
        if i >= 6:
            print('{:7.9f}, '.format(eval_measures_depth[i]), end='')
        else:
            print('{:7.3f}, '.format(eval_measures_depth[i]), end='')
    print('{:7.9f}'.format(eval_measures_depth[8]))

    print('Computing IntegratedReDepth errors for {} eval samples'.format(int(eval_measures_lateralre[-1])))
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>9}, {:>9}, {:>9}".format('silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
    for i in range(8):
        if i >= 6:
            print('{:7.9f}, '.format(eval_measures_lateralre[i]), end='')
        else:
            print('{:7.3f}, '.format(eval_measures_lateralre[i]), end='')
    print('{:7.9f}'.format(eval_measures_lateralre[8]))

    return eval_measures_depth, eval_measures_lateralre

def compute_intre(integrater, normoptizer, outputs, intrinsic, depth_gt, scales=4):
    pred_log = normoptizer.ang2log(intrinsic=intrinsic, ang=outputs[('shape', 0)])
    variance = outputs[('variance', 0)]

    _, _, h, w = depth_gt.shape

    mask = torch.ones_like(depth_gt)
    singularnorm = normoptizer.ang2edge(ang=outputs[('shape', 0)], intrinsic=intrinsic)
    mask = mask * (1 - singularnorm)
    mask[:, :, 0:100, :] = 0
    mask = mask.int().contiguous()

    outputs['intmask'] = mask

    for k in range(scales):
        outputs[('depth', k)] = F.interpolate(outputs[('depth', k)], [h, w], mode='bilinear', align_corners=False)
        outputs[('intre', k)] = integrater.forward(pred_log=pred_log, mask=mask, variance=variance, depthin=outputs[('depth', k)], lam=outputs[('lambda', 0)])

def rename_state_dict(state_dict):
    new_state_dict = dict()
    for key in state_dict.keys():
        if 'module' in key:
            newkey = '.'.join(key.split('.')[1::])
            new_state_dict[newkey] = state_dict[key]
    return new_state_dict

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # Create model
    model = ShapeNet(args)

    normoptizer_eval = SurfaceNormalOptimizer(height=kbcroph, width=kbcropw, batch_size=1, angw=args.angw, vlossw=args.vlossw, sclw=args.sclw)
    normoptizer_eval = normoptizer_eval.cuda()
    crfIntegrater = CRFIntegrationModule(clipvariance=args.clipvariance, maxrange=args.maxrange)
    crfIntegrater = crfIntegrater.cuda()

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(rename_state_dict(checkpoint['model']))
    model.cuda()

    cudnn.benchmark = True
    dataloader_eval = KittiShapeDataLoader(args, 'eval')

    model.eval()
    online_eval(model=model, normoptizer_eval=normoptizer_eval, crfIntegrater=crfIntegrater, dataloader_eval=dataloader_eval, gpu=gpu, ngpus=ngpus_per_node)


if __name__ == '__main__':
    args.distributed = False
    main_worker(args.gpu, 1, args)
