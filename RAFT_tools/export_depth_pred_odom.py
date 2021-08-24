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
import glob

from bts_dataloader import *
from bts import BtsModel

from util import *
from torchvision.transforms import ColorJitter
import torch.multiprocessing as mp

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

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
parser.add_argument('--data_path', type=str, help='path to the data')
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file')
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=80)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='nyu')
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--save_lpg', help='if set, save outputs from lpg layers', action='store_true')
parser.add_argument('--bts_size', type=int, help='initial num_filters in bts', default=512)
parser.add_argument('--usesyncnorm', help='if set, save outputs from lpg layers', action='store_true')
parser.add_argument('--istest', help='if set, save outputs from lpg layers', action='store_true')
parser.add_argument('--evalonly', help='if set, save outputs from lpg layers', action='store_true')

parser.add_argument('--gt_path', type=str)
parser.add_argument('--exportroot', type=str)
parser.add_argument('--odom_root', type=str)
parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=80)

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()


def get_odomentries(args, istrain):
    import glob
    odomentries = list()

    if not istrain:
        odomseqs = [
            '2011_09_30/2011_09_30_drive_0033_sync',
            '2011_09_30/2011_09_30_drive_0034_sync'
        ]
    else:
        odomseqs = [
            '2011_10_03/2011_10_03_drive_0027_sync',
            '2011_10_03/2011_10_03_drive_0042_sync',
            "2011_10_03/2011_10_03_drive_0034_sync",
            "2011_09_26/2011_09_26_drive_0067_sync",
            "2011_09_30/2011_09_30_drive_0016_sync",
            "2011_09_30/2011_09_30_drive_0018_sync",
            "2011_09_30/2011_09_30_drive_0020_sync",
            "2011_09_30/2011_09_30_drive_0027_sync",
            "2011_09_30/2011_09_30_drive_0028_sync"
        ]
    for odomseq in odomseqs:
        if os.path.isdir(os.path.join(args.odom_root, odomseq)):
            tmproot = args.odom_root
        else:
            tmproot = args.data_path
        leftimgs = glob.glob(os.path.join(tmproot, odomseq, 'image_02/data', "*.png"))
        print("%s : img : %d" % (odomseq, len(leftimgs)))
        for leftimg in leftimgs:
            imgname = os.path.basename(leftimg)
            odomentries.append("{} {} {}".format(odomseq, imgname.rstrip('.png'), 'l'))
    return odomentries

def read_splits(args, istrain):
    odom_entries = get_odomentries(args, istrain)
    return odom_entries

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

def get_intrinsic_extrinsic(cam2cam, velo2cam, imu2cam):
    pose_imu2cam = np.eye(4)
    pose_imu2cam[0:3, 0:3] = np.reshape(imu2cam['R'], [3, 3])
    pose_imu2cam[0:3, 3] = imu2cam['T']

    pose_velo2cam = np.eye(4)
    pose_velo2cam[0:3, 0:3] = np.reshape(velo2cam['R'], [3, 3])
    pose_velo2cam[0:3, 3] = velo2cam['T']

    R_rect_00 = np.eye(4)
    R_rect_00[0:3, 0:3] = cam2cam['R_rect_00'].reshape(3, 3)

    intrinsic = np.eye(4)
    intrinsic[0:3, 0:3] = cam2cam['P_rect_02'].reshape(3, 4)[0:3, 0:3]

    org_intrinsic = np.eye(4)
    org_intrinsic[0:3, :] = cam2cam['P_rect_02'].reshape(3, 4)
    extrinsic_from_intrinsic = np.linalg.inv(intrinsic) @ org_intrinsic
    extrinsic_from_intrinsic[0:3, 0:3] = np.eye(3)

    extrinsic = extrinsic_from_intrinsic @ R_rect_00 @ pose_velo2cam @ pose_imu2cam

    return intrinsic.astype(np.float32), extrinsic.astype(np.float32)

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims):
        self.wd, self.ht = dims
        dsrat = 32
        pad_ht = (((self.ht // dsrat) + 1) * dsrat - self.ht) % dsrat
        pad_wd = (((self.wd // dsrat) + 1) * dsrat - self.wd) % dsrat
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, image):
        return F.pad(image, self._pad, mode='replicate')

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def remove_dup(entries):
    dupentry = list()
    for entry in entries:
        seq, index, _ = entry.split(' ')
        dupentry.append("{} {}".format(seq, index.zfill(10)))

    removed = list(set(dupentry))
    removed.sort()
    return removed

def export(gpuid, model, args, ngpus_per_node, evaluation_entries, istrain=False):
    """Test function."""
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    args.mode = 'test'
    args.gpu = gpuid
    args.dist_backend = 'nccl'
    args.dist_url = 'tcp://127.0.0.1:1234'
    args.world_size = ngpus_per_node
    args.rank = gpuid

    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    model.eval()

    loc = 'cuda:{}'.format(args.gpu)
    checkpoint = torch.load(args.checkpoint_path, map_location=loc)
    model.load_state_dict(checkpoint['model'])

    interval = np.floor(len(evaluation_entries) / ngpus_per_node).astype(np.int).item()
    if gpuid == ngpus_per_node - 1:
        stidx = int(interval * gpuid)
        edidx = len(evaluation_entries)
    else:
        stidx = int(interval * gpuid)
        edidx = int(interval * (gpuid + 1))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if istrain:
        jitterparam = 0.95
        photo_aug = ColorJitter(brightness=jitterparam, contrast=jitterparam, saturation=jitterparam, hue=jitterparam / 3.14)

    print("Initialize Instance on Gpu %d, from %d to %d, total %d" % (gpuid, stidx, edidx, len(evaluation_entries)))

    with torch.no_grad():
        for t_idx, entry in enumerate(tqdm(evaluation_entries[stidx : edidx])):
            seq, index = entry.split(' ')
            torch.manual_seed(int(t_idx))

            export_fold = os.path.join(args.exportroot, seq, 'image_02')
            os.makedirs(export_fold, exist_ok=True)
            export_path = os.path.join(export_fold, "{}.png".format(str(index).zfill(10)))

            imgpath = os.path.join(args.data_path, seq, 'image_02', 'data', "{}.png".format(str(index).zfill(10)))
            if os.path.exists(imgpath):
                tmproot = args.data_path
            else:
                tmproot = args.odom_root
            imgpath = os.path.join(tmproot, seq, 'image_02', 'data', "{}.png".format(str(index).zfill(10)))

            calib_dir = os.path.join(tmproot, seq.split('/')[0])

            cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
            velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
            imu2cam = read_calib_file(os.path.join(calib_dir, 'calib_imu_to_velo.txt'))
            intrinsic, extrinsic = get_intrinsic_extrinsic(cam2cam, velo2cam, imu2cam)

            image = Image.open(imgpath)
            padder = InputPadder(image.size)

            if istrain:
                image = photo_aug(image)
            image = np.asarray(image, dtype=np.float64) / 255.0
            image = normalize(torch.from_numpy(image).permute([2, 0, 1])).unsqueeze(0)
            image = padder.pad(image)
            image = Variable(image.float().cuda(gpuid))

            focal = Variable(torch.from_numpy(np.array(intrinsic[0, 0])).unsqueeze(0).cuda(gpuid))

            # Predict
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)
            depth_est = padder.unpad(depth_est)
            depth_est = depth_est.squeeze().cpu().numpy()
            depth_est = (depth_est * 256.0).astype(np.uint16)

            Image.fromarray(depth_est).save(export_path)

    return

def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return lines

def evaluation():
    evaluation_entries = get_num_lines(args.filenames_file)
    print('now testing {} files with {}'.format(len(evaluation_entries), args.checkpoint_path))

    metrics = list()
    countnum = 0
    for t_idx, sample in enumerate(tqdm(evaluation_entries)):
        gt_depth_path = os.path.join(args.gt_path, evaluation_entries[t_idx].split(' ')[1])
        pred_depth_path = os.path.join(args.exportroot, evaluation_entries[t_idx].split(' ')[1])
        if os.path.exists(gt_depth_path):
            gt_depth = cv2.imread(gt_depth_path, -1)
            gt_depth = gt_depth.astype(np.float32) / 256.0

            # Predict
            pred_depth = cv2.imread(pred_depth_path, -1)
            pred_depth = pred_depth.astype(np.float32) / 256.0

            pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
            pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
            pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
            pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)
            eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

            metric = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
            metrics.append(metric)
            countnum += 1
        else:
            print("%s missing" % gt_depth_path)
    metrics = np.stack(metrics, axis=0)
    metrics = np.mean(metrics, axis=0)

    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'log10', 'abs_rel', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
    for i in range(8):
        print('{:7.3f}, '.format(metrics[i]), end='')
    print('{:7.3f}'.format(metrics[8]))

if __name__ == '__main__':
    ngpus_per_node = torch.cuda.device_count()
    if args.evalonly:
        evaluation()
    else:
        model = BtsModel(params=args)
        train_entries = read_splits(args, istrain=True)
        mp.spawn(export, nprocs=ngpus_per_node, args=(model, args, ngpus_per_node, remove_dup(train_entries), True))
        eval_entries = read_splits(args, istrain=False)
        mp.spawn(export, nprocs=ngpus_per_node, args=(model, args, ngpus_per_node, remove_dup(eval_entries), False))
        evaluation()
