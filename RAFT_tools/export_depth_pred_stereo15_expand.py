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

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
parser.add_argument('--istest', help='if set, save outputs from lpg layers', action='store_true')
parser.add_argument('--trainonly', help='if set, save outputs from lpg layers', action='store_true')

parser.add_argument('--gt_path', type=str)
parser.add_argument('--exportroot', type=str)
parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--mpf_root', type=str)

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

def readlines(filename):
    with open(filename, 'r') as f:
        filenames = f.readlines()
    return filenames

def read_splits_mapping(args):
    evaluation_entries = []
    import glob
    for m in range(200):
        seqname = "kittistereo15_{}/kittistereo15_{}_sync".format(str(m).zfill(6), str(m).zfill(6))
        evaluation_entries.append("{} {} {}".format(seqname, "10".zfill(10), 'l'))

    expandentries = list()
    mappings = readlines(args.mpf_root)
    for idx, m in enumerate(mappings):
        if len(m) == 1:
            continue
        d, s, cidx = m.split(' ')
        seq = "{}/{}".format(d, s)
        pngs = glob.glob(os.path.join(args.data_path, d, s, 'image_02/data', '*.png'))
        for p in pngs:
            frmidx = p.split('/')[-1].split('.')[0]
            expandentries.append("{} {} l".format(seq, frmidx.zfill(10)))
    expandentries = list(set(expandentries))
    expandentries.sort()
    return expandentries

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

def export(args, evaluation_entries):
    """Test function."""
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    args.mode = 'test'
    args.num_threads = 0

    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    print('now testing {} files with {}'.format(len(evaluation_entries), args.checkpoint_path))

    with torch.no_grad():
        for t_idx, entry in enumerate(tqdm(evaluation_entries)):
            seq, index, _ = entry.split(' ')

            imgpath = os.path.join(args.data_path, seq, 'image_02', 'data', "{}.png".format(str(index).zfill(10)))
            calib_dir = os.path.join(args.data_path, seq.split('/')[0])

            cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
            velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
            imu2cam = read_calib_file(os.path.join(calib_dir, 'calib_imu_to_velo.txt'))
            intrinsic, extrinsic = get_intrinsic_extrinsic(cam2cam, velo2cam, imu2cam)

            image = Image.open(imgpath)
            padder = InputPadder(image.size)

            image = np.asarray(image, dtype=np.float64) / 255.0
            image = normalize(torch.from_numpy(image).permute([2, 0, 1])).unsqueeze(0)
            image = padder.pad(image)
            image = Variable(image.float().cuda())

            focal = Variable(torch.from_numpy(np.array(intrinsic[0, 0])).unsqueeze(0).cuda())

            # Predict
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)
            depth_est = padder.unpad(depth_est)
            depth_est = depth_est.squeeze().cpu().numpy()
            depth_est = (depth_est * 256.0).astype(np.uint16)

            export_fold = os.path.join(args.exportroot, seq, 'image_02')
            os.makedirs(export_fold, exist_ok=True)
            export_path = os.path.join(export_fold, "{}.png".format(str(index).zfill(10)))
            Image.fromarray(depth_est).save(export_path)

    return


if __name__ == '__main__':
    eval_entries = read_splits_mapping(args)
    export(args, eval_entries)