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

from Exp_Shapenet_monodepth.shapedataset import KittiShapeDataLoader, KittiShapeDataset
from util import *
from Exp_Shapenet_monodepth.shapenet import ShapeNet
from torchvision import transforms
import torch.utils.data.distributed
import torch.distributed as dist

parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')

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
parser.add_argument('--vlossw',                    type=float, default=0.2)
parser.add_argument('--sclw',                      type=float, default=0)

# Preprocessing
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Multi-gpu training
parser.add_argument('--num_workers',               type=int,   help='number of threads to use for data loading', default=1)
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

kbcroph = 352
kbcropw = 1216

def block_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__

def online_eval(model, normoptizer_eval, dataloader_eval, gpu, ngpus):
    eval_measures = torch.zeros(2).cuda(device=gpu)
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(args.gpu, non_blocking=True))
            K = torch.autograd.Variable(eval_sample_batched['K'].cuda(gpu, non_blocking=True))
            depth_gt = torch.autograd.Variable(eval_sample_batched['depth'].cuda(args.gpu, non_blocking=True))

            if 'depth' not in eval_sample_batched:
                continue

            pred_shape = model(image)
            pred_shape = F.interpolate(pred_shape, [kbcroph, kbcropw], mode='bilinear', align_corners=True)
            loss = normoptizer_eval.intergrationloss_ang_validation(ang=pred_shape, intrinsic=K, depthMap=depth_gt)

        eval_measures[0] += loss
        eval_measures[1] += 1

    eval_measures_cpu = eval_measures.cpu()
    L1Measure = float(eval_measures_cpu[0] / eval_measures_cpu[1])
    print('L1 Loss: %f, from %d eval samples' % (L1Measure, int(eval_measures_cpu[1])))
    return L1Measure

def readlines(filename):
    with open(filename, 'r') as f:
        filenames = f.readlines()
    return filenames

def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass
    return data

def main_worker():
    normoptizer_eval = SurfaceNormalOptimizer(height=kbcroph, width=kbcropw, batch_size=1)

    shapepredroot = '/home/shengjie/Documents/Data/Kitti/kitti_angpred'
    semigtroot = '/home/shengjie/Documents/Data/Kitti/semidense_gt'
    kittiroot = '/home/shengjie/Documents/Data/Kitti/kitti_raw/kitti_data'
    split_test = '/home/shengjie/Documents/Project_SemanticDepth/splits/eigen/test_files.txt'
    entries = readlines(split_test)

    losses = list()
    for entry in entries:
        seq, frame, dir = entry.split(' ')

        if not os.path.exists(os.path.join(semigtroot, seq, 'image_02', "{}.png".format(frame.zfill(10)))):
            continue

        depthgt = pil.open(os.path.join(semigtroot, seq, 'image_02', "{}.png".format(frame.zfill(10))))
        width, height = depthgt.size
        top_margin = int(height - kbcroph)
        left_margin = int((width - kbcropw) / 2)

        depthgt = depthgt.crop((left_margin, top_margin, left_margin + kbcropw, top_margin + kbcroph))
        depthgt = np.array(depthgt).astype(np.float32) / 256.0
        depthgt = torch.from_numpy(depthgt).unsqueeze(0).unsqueeze(0).float()

        angh = pil.open(os.path.join(shapepredroot, "angh", seq, 'image_02', str(frame).zfill(10) + '.png'))
        angh = angh.crop((left_margin, top_margin, left_margin + kbcropw, top_margin + kbcroph))
        angv = pil.open(os.path.join(shapepredroot, "angv", seq, 'image_02', str(frame).zfill(10) + '.png'))
        angv = angv.crop((left_margin, top_margin, left_margin + kbcropw, top_margin + kbcroph))

        angh = np.array(angh).astype(np.float32)
        angh = (angh / 255.0 / 255.0 - 0.5) * 2 * np.pi
        angh = torch.from_numpy(angh).unsqueeze(0).unsqueeze(0).float()
        angv = np.array(angv).astype(np.float32)
        angv = (angv / 255.0 / 255.0 - 0.5) * 2 * np.pi
        angv = torch.from_numpy(angv).unsqueeze(0).unsqueeze(0).float()

        angpred = torch.cat([angh, angv], dim=1)
        cam2cam = read_calib_file(os.path.join(kittiroot, seq.split('/')[0], 'calib_cam_to_cam.txt'))
        K = np.eye(4)
        K[0:3, :] = cam2cam['P_rect_0{}'.format(str(2))].reshape(3, 4)
        K[0, 2] = K[0, 2] - left_margin
        K[1, 2] = K[1, 2] - top_margin
        K = torch.from_numpy(K).unsqueeze(0).float()

        losses.append(normoptizer_eval.intergrationloss_ang_validation(ang=angpred, intrinsic=K, depthMap=depthgt))

    print('L1 Loss: %f' % (np.array(losses).mean()))

def main():
    main_worker()


if __name__ == '__main__':
    main()
