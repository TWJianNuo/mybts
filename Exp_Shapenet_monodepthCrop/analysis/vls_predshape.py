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
import argparse
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__))) # location of src
from util import *
from Exp_Shapenet_monodepthCrop.shapenet import ShapeNet
import torch.utils.data.distributed
import torch.distributed as dist

import transforms as T
from transforms.augmentation import _transform_to_aug
from fvcore.transforms.transform import CropTransform
import torchvision
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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

losses = list()

def main_worker():
    # Create empty argument
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.bts_size = 512
    args.expname = 'shapenet_320x1024_monodepthCrop'
    args.checkpoint_path = '/home/shengjie/Documents/bts/tmp/{}/model-72500-best_L1Measure_0.00706'.format(args.expname)

    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:{}'.format(np.random.randint(1111, 9999), 1), world_size=1, rank=0)

    model = ShapeNet()
    checkpoint = torch.load(args.checkpoint_path)
    for key in checkpoint['model'].keys():
        if 'module' in key:
            model = torch.nn.DataParallel(model)
            break
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(torch.device("cuda"))

    kbcroph = 352
    kbcropw = 1216

    normoptizer_eval = SurfaceNormalOptimizer(height=kbcroph, width=kbcropw, batch_size=1).cuda()

    testfilepath = '/home/shengjie/Documents/Project_SemanticDepth/splits/eigen_full/test_files.txt'
    entries = readlines(testfilepath)

    kittiroot = '/home/shengjie/Documents/Data/Kitti/kitti_raw/kitti_data'
    filterlidar_gtroot = '/home/shengjie/Documents/Data/Kitti/filtered_lidar'
    vlsroot = '/media/shengjie/disk1/visualization/bts_shape_pred'
    vlsroot = os.path.join(vlsroot, args.expname)

    os.makedirs(vlsroot, exist_ok=True)

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    minang = - np.pi / 3 * 2
    maxang = 2 * np.pi - np.pi / 3 * 2
    with torch.no_grad():
        for idx, entry in enumerate(entries):
            seq, index, dir = entry.split(' ')

            if not os.path.exists(os.path.join(filterlidar_gtroot, seq, 'image_02', "{}.png".format(index))):
                continue

            rgb = pil.open(os.path.join(kittiroot, seq, 'image_02', "data", "{}.png".format(index)))
            rgb = np.array(rgb).astype(np.float) / 255.0
            depthgt = pil.open(os.path.join(filterlidar_gtroot, seq, 'image_02', "{}.png".format(index)))
            depthgt = np.array(depthgt).astype(np.float) / 256.0

            calibpath = os.path.join(kittiroot, seq.split('/')[0], 'calib_cam_to_cam.txt')
            cam2cam = read_calib_file(calibpath)
            K = np.eye(4, dtype=np.float32)
            K[0:3, :] = cam2cam['P_rect_02'].reshape(3, 4)

            height, width, _ = rgb.shape

            top_margin = int(height - kbcroph)
            left_margin = int((width - kbcropw) / 2)

            kbcrop_transform = CropTransform(left_margin, top_margin, kbcropw, kbcroph)
            kbcrop_aug = _transform_to_aug(kbcrop_transform)
            aug_input = T.AugInput(rgb, intrinsic=K)
            transforms = kbcrop_aug(aug_input)
            rgb, K = aug_input.image, aug_input.intrinsic
            depthgt = transforms.apply_image(depthgt)

            rgb = normalize(torch.from_numpy(rgb).permute([2, 0, 1])).unsqueeze(0).float()
            depthgt = torch.from_numpy(depthgt).unsqueeze(0).unsqueeze(0).float()
            K = torch.from_numpy(K).unsqueeze(0).float()

            rgb = torch.autograd.Variable(rgb.cuda())
            depthgt = torch.autograd.Variable(depthgt.cuda())
            K = torch.autograd.Variable(K.cuda())

            pred_shape = model(rgb)

            pred_norm = normoptizer_eval.ang2normal(ang=pred_shape, intrinsic=K)

            angh = pred_shape[:, 0].unsqueeze(1)
            angv = pred_shape[:, 1].unsqueeze(1)

            fig_angh = tensor2disp(angh - minang, vmax=maxang, viewind=0)
            fig_angv = tensor2disp(angv - minang, vmax=maxang, viewind=0)
            fig_norm = tensor2rgb((pred_norm + 1) / 2, isnormed=False, viewind=0)
            fig_combined = np.concatenate([np.array(fig_angh), np.array(fig_angv), np.array(fig_norm)], axis=1)

            losses.append(float(normoptizer_eval.intergrationloss_ang_validation(ang=pred_shape, intrinsic=K, depthMap=depthgt).cpu().numpy()))

            # figname = "{}_{}.png".format(seq.split('/')[1], str(index).zfill(10))
            # pil.fromarray(fig_combined).save(os.path.join(vlsroot, figname))
            #
            print("Finished: %d" % idx)

        L1measure = np.array(losses).mean()
        performancetxt = "L1_{:.5f}.txt".format(L1measure)
        performancetxt = open(os.path.join(vlsroot, performancetxt), "w")
        performancetxt.close()

        print("L1 Measure: %f" % L1measure)


if __name__ == '__main__':
    main_worker()
