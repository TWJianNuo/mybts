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
from PIL import Image
import os
import random

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

class BtsDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler, drop_last=True)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        if mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        K, focal = self.get_intrinsic(os.path.join(self.args.data_path, sample_path.split(' ')[0].split('/')[0], 'calib_cam_to_cam.txt'))

        if self.mode == 'train':
            do_flip = random.random() > 0.5
        else:
            do_flip = False

        inputs = dict()

        inputs['image'] = self.get_rgb(os.path.join(self.args.data_path, sample_path.split()[0]), do_flip)

        if sample_path.split()[1] != 'None':
            inputs['depth'] = self.get_depth(os.path.join(self.args.gt_path, sample_path.split()[1]), do_flip)
            gtshape = inputs['depth'].size
        if self.args.inslabel_path is not None:
            inputs['inslabel'] = self.get_ins(sample_path, do_flip)
        if self.args.angdata_path is not None:
            inputs.update(self.get_angs(sample_path, do_flip))

        if self.args.do_kb_crop is True:
            inputs, K = self.do_kb_crop(inputs, K)

        if self.mode == 'train':
            inputs, K = self.random_crop(inputs, K)
            inputs = self.img2np(inputs)
            inputs['image'] = self.train_preprocess(inputs['image'])
            inputs['K'] = K
            inputs['focal'] = focal
        else:
            inputs = self.img2np(inputs)
            inputs['K'] = K
            inputs['focal'] = focal
            inputs['gtshape'] = np.array(gtshape)

        if self.transform:
            sample = self.transform(inputs)

        return sample

    def img2np(self, inputs):
        inputs['image'] = np.asarray(inputs['image'], dtype=np.float32) / 255.0
        if 'depth' in inputs:
            inputs['depth'] = np.expand_dims(np.asarray(inputs['depth'], dtype=np.float32), axis=2) / 256.0

        if 'angh' in inputs and 'angv' in inputs:
            inputs['angh'] = np.expand_dims((np.asarray(inputs['angh'], dtype=np.float32) / 255.0 / 255.0 - 0.5) * 2 * np.pi, axis=2)
            inputs['angv'] = np.expand_dims((np.asarray(inputs['angv'], dtype=np.float32) / 255.0 / 255.0 - 0.5) * 2 * np.pi, axis=2)

        if 'inslabel' in inputs:
            inputs['inslabel'] = np.expand_dims(np.asarray(inputs['inslabel'], dtype=np.int32), axis=2)

        return inputs

    def do_kb_crop(self, inputs, K):
        kbcroph = 352
        kbcropw = 1216

        width, height = inputs['image'].size
        top_margin = int(height - kbcroph)
        left_margin = int((width - kbcropw) / 2)

        for k in inputs.keys():
            sz = inputs[k].size
            assert sz[0] == width and sz[1] == height, print("Image Shape Dismatch")
            inputs[k] = inputs[k].crop((left_margin, top_margin, left_margin + kbcropw, top_margin + kbcroph))

        K[0, 2] = K[0, 2] - left_margin
        K[1, 2] = K[1, 2] - top_margin

        return inputs, K

    def random_crop(self, inputs, K):
        x = random.randint(0, inputs['image'].size[0] - self.args.input_width)
        y = random.randint(0, inputs['image'].size[1] - self.args.input_height)
        for k in inputs.keys():
            sz = inputs[k].size
            assert sz[0] >= self.args.input_width and sz[1] >= self.args.input_height, print("Crop size larger than image size")
            inputs[k] = inputs[k].crop((x, y, x + self.args.input_width, y + self.args.input_height))
        K[0, 2] = K[0, 2] - x
        K[1, 2] = K[1, 2] - y
        return inputs, K

    def get_ins(self, sample_path, do_flip):
        inspath = os.path.join(self.args.inslabel_path, sample_path.split()[0].replace('/data', ''))
        inslabel = Image.open(inspath)
        if do_flip:
            inslabel = inslabel.transpose(Image.FLIP_LEFT_RIGHT)
        return inslabel

    def get_angs(self, sample_path, do_flip):
        inputs = dict()
        if do_flip:
            angh_path = os.path.join(self.args.angdata_path, 'angh_flipped', sample_path.split()[0].replace('/data', ''))
            angv_path = os.path.join(self.args.angdata_path, 'angv_flipped', sample_path.split()[0].replace('/data', ''))
        else:
            angh_path = os.path.join(self.args.angdata_path, 'angh', sample_path.split()[0].replace('/data', ''))
            angv_path = os.path.join(self.args.angdata_path, 'angv', sample_path.split()[0].replace('/data', ''))

        angh = Image.open(angh_path)
        angv = Image.open(angv_path)

        inputs['angh'] = angh
        inputs['angv'] = angv
        return inputs

    def get_depth(self, depth_path, do_flip):
        depth_gt = Image.open(depth_path)
        if do_flip:
            depth_gt = depth_gt.transpose(Image.FLIP_LEFT_RIGHT)
        return depth_gt

    def get_rgb(self, image_path, do_flip):
        image = Image.open(image_path)
        if do_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    def get_intrinsic(self, calibpath):
        cam2cam = read_calib_file(calibpath)
        K = np.eye(4)
        K[0:3, :] = cam2cam['P_rect_02'].reshape(3, 4)
        return K, K[0, 0]

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def train_preprocess(self, image):
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image

    def augment_image(self, image):
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
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        sample['image'] = self.normalize(torch.from_numpy(sample['image']).permute([2,0,1]).contiguous().float())
        sample['K'] = torch.from_numpy(sample['K']).float()

        if 'inslabel' in sample:
            sample['inslabel'] = torch.from_numpy(sample['inslabel']).permute([2,0,1]).contiguous().int()

        if 'angh' in sample and 'angv' in sample:
            sample['angh'] = torch.from_numpy(sample['angh']).permute([2,0,1]).contiguous().float()
            sample['angv'] = torch.from_numpy(sample['angv']).permute([2,0,1]).contiguous().float()

        if 'depth' in sample:
            sample['depth'] = torch.from_numpy(sample['depth']).permute([2, 0, 1]).contiguous().float()

        if 'gtshape' in sample:
            sample['gtshape'] = torch.from_numpy(sample['gtshape']).contiguous().float()
        return sample