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
import torch
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

class BtsDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = KittiDataset(args, mode, filenames=readlines(args.filenames_file))
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples) if args.distributed else None
            self.data = DataLoader(self.training_samples, args.batch_size, shuffle=(self.train_sampler is None), num_workers=args.num_threads, pin_memory=True, sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = KittiDataset(args, mode, filenames=readlines(args.filenames_file_eval))
            self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False) if args.distributed else None
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=args.num_threads_eval, pin_memory=True, sampler=self.eval_sampler)
        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


class KittiDataset(Dataset):
    def __init__(self, args, mode, filenames):
        self.args = args
        self.filenames = filenames
        self.mode = mode
        self.color_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.filter_file()

        if mode == 'train':
            self.data_path = self.args.data_path
            self.gt_path = self.args.gt_path
            self.crop_flip = T.AugmentationList([
                T.RandomCrop(crop_type='absolute', crop_size=[self.args.input_height, self.args.input_width]),
                T.RandomFlip(prob=0.5, horizontal=True)])

        elif mode == 'online_eval':
            self.data_path = self.args.data_path_eval
            self.gt_path = self.args.gt_path_eval

    def filter_file(self):
        filteredfilenames = list()
        for entry in self.filenames:
            if entry.split(' ')[1] != 'None':
                filteredfilenames.append(entry)
        self.filenames = filteredfilenames

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]

        # Read Intrinsic
        K = self.get_intrinsic(os.path.join(self.data_path, sample_path.split(' ')[0].split('/')[0], 'calib_cam_to_cam.txt'))

        # Read RGB
        image = self.get_rgb(os.path.join(self.data_path, sample_path.split()[0]))

        # Read Depth
        depth_gt = self.get_depth(os.path.join(self.gt_path, sample_path.split()[1]))

        # Read focal
        focal = float(sample_path.split()[2])

        # Read Size
        size = np.array(image.size)

        # Convert to numpy
        image, depth_gt = self.cvt_np(image=image, depth_gt=depth_gt)

        if self.args.do_kb_crop:
            image, depth_gt, K = self.do_kb_crop(image=image, K=K, depth_gt=depth_gt)

        # Augmentation in Training
        if self.mode == 'train':
            # Random Rotation
            if self.args.do_random_rotate:
                image, depth_gt = self.do_random_rotate(image=image, depth_gt=depth_gt)

            # Random Crop and Flip
            image, depth_gt, K = self.do_random_crop_flip(image=image, depth_gt=depth_gt, K=K)
            image = self.do_random_coloraug(image=image)

        image, depth_gt, K = self.to_tensor(image=image, depth_gt=depth_gt, K=K)
        image = self.color_normalize(image)
        sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'K': K, 'size': size}
        return sample

    def get_depth(self, depth_path):
        depth_gt = Image.open(depth_path)
        return depth_gt

    def get_rgb(self, image_path):
        image = Image.open(image_path)
        return image

    def get_intrinsic(self, calibpath):
        cam2cam = read_calib_file(calibpath)
        K = np.eye(4)
        K[0:3, :] = cam2cam['P_rect_02'].reshape(3, 4)
        return K

    def do_kb_crop(self, image, depth_gt, K):
        kbcroph = 352
        kbcropw = 1216

        height, width, _ = image.shape

        top_margin = int(height - kbcroph)
        left_margin = int((width - kbcropw) / 2)

        kbcrop_aug = _transform_to_aug(CropTransform(left_margin, top_margin, kbcropw, kbcroph))
        aug_input = T.AugInput(image, intrinsic=K)

        transforms = kbcrop_aug(aug_input)

        image, K = aug_input.image, aug_input.intrinsic
        depth_gt = transforms.apply_image(depth_gt)

        return image, depth_gt, K

    def do_random_rotate(self, image, depth_gt):
        random_angle = (random.random() - 0.5) * 2 * self.args.degree
        image = Image.fromarray(image).rotate(random_angle, Image.BILINEAR)
        depth_gt = Image.fromarray(depth_gt).rotate(random_angle, Image.NEAREST)
        return np.array(image), np.array(depth_gt)

    def do_random_crop_flip(self, image, depth_gt, K):
        aug_input = T.AugInput(image, intrinsic=K)
        transforms = self.crop_flip(aug_input)
        image, K = aug_input.image, aug_input.intrinsic
        depth_gt = transforms.apply_image(depth_gt)
        return image, depth_gt, K

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def do_random_coloraug(self, image):
        image = image.astype(np.float32) / 255.0
        if random.random() > 0.5:
            # gamma augmentation
            gamma = random.uniform(0.9, 1.1)
            image_aug = image ** gamma

            # brightness augmentation
            brightness = random.uniform(0.9, 1.1)
            image_aug = image_aug * brightness

            # color augmentation
            colors = np.random.uniform(0.9, 1.1, size=3)
            white = np.ones((image.shape[0], image.shape[1]))
            color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
            image_aug *= color_image
            image_aug = np.clip(image_aug, 0, 1) * 255.0
        else:
            image_aug = image * 255.0
        return image_aug

    def cvt_np(self, image, depth_gt):
        image = np.array(image)
        depth_gt = np.array(depth_gt)
        return image, depth_gt

    def to_tensor(self, image, depth_gt, K):
        image = torch.from_numpy(np.copy(image)).permute([2, 0, 1]).contiguous().float() / 255.0
        depth_gt = torch.from_numpy(np.copy(depth_gt)).unsqueeze(0).contiguous().float() / 256.0
        K = torch.from_numpy(K).float()
        return image, depth_gt, K

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)

        depth = sample['depth']
        depth = self.to_tensor(depth)
        return {'image': image, 'depth': depth, 'focal': focal}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img