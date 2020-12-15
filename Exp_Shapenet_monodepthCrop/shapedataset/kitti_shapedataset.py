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

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms
from PIL import Image, ImageFile
import os
import random

import transforms as T
from transforms.augmentation import _transform_to_aug
from fvcore.transforms.transform import CropTransform
ImageFile.LOAD_TRUNCATED_IMAGES = True

from distributed_sampler_no_evenly_divisible import *


def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])

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

def readlines(filename):
    with open(filename, 'r') as f:
        filenames = f.readlines()
    return filenames

class KittiShapeDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = KittiShapeDataset(args, mode, filenames=readlines(args.filenames_file))

            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None
            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_workers,
                                   pin_memory=True,
                                   sampler=self.train_sampler, drop_last=True)

        elif mode == 'eval':
            self.testing_samples = KittiShapeDataset(args, mode, filenames=readlines(args.filenames_file_eval))

            if args.distributed:
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=0,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)
        else:
            print('mode should be one of \'train, eval\'. Got {}'.format(mode))


class KittiShapeDataset(Dataset):
    def __init__(self, args, mode, filenames):
        self.args = args

        self.filenames = filenames
        self.mode = mode
        self.to_tensor = ToTensor

        self.filter_file()

        if self.mode == 'train':
            self.valuegaugs = T.AugmentationList([
                T.RandomCrop(crop_type='absolute', crop_size=[self.args.input_height, self.args.input_width]),
                T.RandomFlip(prob=0.5, horizontal=True)])
        self.toTensor = ToTensor()

    def filter_file(self):
        filteredfilenames = list()
        for entry in self.filenames:
            if entry.split(' ')[1] != 'None':
                filteredfilenames.append(entry)
        self.filenames = filteredfilenames

    def __getitem__(self, idx):
        inputs = dict()

        sample_path = self.filenames[idx]

        # Read Intrinsic
        K = self.get_intrinsic(os.path.join(self.args.data_path, sample_path.split(' ')[0].split('/')[0], 'calib_cam_to_cam.txt'))

        # Read RGB
        image = self.get_rgb(os.path.join(self.args.data_path, sample_path.split()[0]))

        # Read Depth
        depthgt = self.get_depth(os.path.join(self.args.gt_path, sample_path.split()[1]))

        if self.mode == 'train':
            # Do Crop and Flip Augmentations
            aug_input = T.AugInput(image, intrinsic=K)
            transforms = self.valuegaugs(aug_input)
            image, K = aug_input.image, aug_input.intrinsic
            depthgt = transforms.apply_image(depthgt)
            image = self.augment_image(image)

            # self.test_augmentations(sample_path, image, depthgt, K, transforms)
        elif self.mode == 'eval':
            # Do Kb crop
            kbcroph = 352
            kbcropw = 1216

            height, width, _ = image.shape

            top_margin = int(height - kbcroph)
            left_margin = int((width - kbcropw) / 2)

            kbcrop_transform = CropTransform(left_margin, top_margin, kbcropw, kbcroph)
            kbcrop_aug = _transform_to_aug(kbcrop_transform)
            aug_input = T.AugInput(image, intrinsic=K)
            transforms = kbcrop_aug(aug_input)
            image, K = aug_input.image, aug_input.intrinsic
            depthgt = transforms.apply_image(depthgt)

        inputs['image'] = image
        inputs['depth'] = depthgt
        inputs['K'] = K

        inputs = self.toTensor(inputs)

        return inputs

    def get_depth(self, depth_path):
        depth_gt = Image.open(depth_path)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256.0
        return depth_gt

    def get_rgb(self, image_path):
        image = Image.open(image_path)
        image = np.array(image).astype(np.float32) / 255.0
        return image

    def get_intrinsic(self, calibpath):
        cam2cam = read_calib_file(calibpath)
        K = np.eye(4)
        K[0:3, :] = cam2cam['P_rect_02'].reshape(3, 4)
        return K

    def augment_image(self, image):
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def test_augmentations(self, sample_path, image_aug, depth_aug, intrinsic_aug, transforms):
        # Test for intrinsic correctness
        intrinsic_org = self.get_intrinsic(os.path.join(self.args.data_path, sample_path.split(' ')[0].split('/')[0], 'calib_cam_to_cam.txt'))
        depth_org = self.get_depth(os.path.join(self.args.gt_path, sample_path.split()[1]))

        from fvcore.transforms.transform import CropTransform, HFlipTransform

        h, w = depth_org.shape
        xx, yy = np.meshgrid(range(w), range(h), indexing="xy")
        selector = np.zeros_like(depth_org)
        selector[transforms[0].y0:transforms[0].y0 + transforms[0].h, transforms[0].x0:transforms[0].x0 + transforms[0].w] = 1
        selector = selector * (depth_org > 0)
        xxs = xx[selector == 1]
        yys = yy[selector == 1]
        rndind = np.random.randint(0, len(xxs))

        tx = xxs[rndind]
        ty = yys[rndind]
        td = depth_org[int(ty), int(tx)]

        ctx = tx - transforms[0].x0
        cty = ty - transforms[0].y0
        if isinstance(transforms[1], HFlipTransform):
            ctd = depth_aug[int(cty), transforms[1].width - 1 - int(ctx)]
        else:
            ctd = depth_aug[int(cty), int(ctx)]

        assert td == ctd

        ptsorg = np.array([[tx * td, ty * td, td, 1]]).T
        ptsorg = np.linalg.inv(intrinsic_org) @ ptsorg

        ptscrop = np.array([[ctx * ctd, cty * ctd, ctd, 1]]).T
        ptscrop = np.linalg.inv(intrinsic_aug) @ ptscrop

        assert np.abs(ptsorg - ptscrop).max() < 1e-4

        # Check for consistency
        import matplotlib.pyplot as plt
        cm = plt.get_cmap('jet')
        crph, crpw = depth_aug.shape
        vlsxx, vlsyy = np.meshgrid(range(crpw), range(crph), indexing='xy')
        vlsselector = depth_aug > 0
        vlsxx = vlsxx[vlsselector]
        vlsyy = vlsyy[vlsselector]
        colors = cm(6 / depth_aug[vlsselector])

        plt.figure()
        plt.imshow(Image.fromarray(image_aug.astype(np.uint8)))
        plt.scatter(vlsxx, vlsyy, 1, colors)


    def __len__(self):
        return len(self.filenames)

class ToTensor(object):
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        sample['image'] = self.normalize(torch.from_numpy(np.copy(sample['image'])).permute([2, 0, 1]).contiguous().float())
        sample['K'] = torch.from_numpy(sample['K']).float()
        if 'depth' in sample:
            sample['depth'] = torch.from_numpy(np.copy(sample['depth'])).unsqueeze(0).contiguous().float()
        return sample