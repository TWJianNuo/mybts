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
                                   sampler=self.train_sampler)

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
        calibpath = os.path.join(self.args.data_path, sample_path.split(' ')[0].split('/')[0], 'calib_cam_to_cam.txt')
        K_org = get_intrinsic(calibpath, camind=2) # We do not use right camera
        focal = K_org[0, 0]
        K_cropped = np.copy(K_org)

        if self.mode == 'train':
            # We do not use the right image here
            do_flip = random.random() > 0.5
            image_path = os.path.join(self.args.data_path, sample_path.split()[0])
            depth_path = os.path.join(self.args.gt_path, sample_path.split()[1])
            if do_flip:
                shapeh_path = os.path.join(self.args.angdata_path, 'angh_flipped', sample_path.split()[0].replace('/data', ''))
                shapev_path = os.path.join(self.args.angdata_path, 'angv_flipped', sample_path.split()[0].replace('/data', ''))
            else:
                shapeh_path = os.path.join(self.args.angdata_path, 'angh', sample_path.split()[0].replace('/data', ''))
                shapev_path = os.path.join(self.args.angdata_path, 'angv', sample_path.split()[0].replace('/data', ''))

            image = Image.open(image_path)
            depth_gt = Image.open(depth_path)
            shapeh = Image.open(shapeh_path)
            shapev = Image.open(shapev_path)

            # Random flipping
            if do_flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                depth_gt = depth_gt.transpose(Image.FLIP_LEFT_RIGHT)

            if self.args.do_kb_crop is True:
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

                shapeh = shapeh.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                shapev = shapev.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

                K_cropped[0, 2] = K_cropped[0, 2] - left_margin
                K_cropped[1, 2] = K_cropped[1, 2] - top_margin

            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            shapeh = np.asarray(shapeh, dtype=np.float32) / 255.0 / 255.0
            shapev = np.asarray(shapev, dtype=np.float32) / 255.0 / 255.0
            shapeh = np.expand_dims(shapeh, axis=2)
            shapev = np.expand_dims(shapev, axis=2)

            depth_gt = depth_gt / 256.0

            image, depth_gt, shapeh, shapev, K_cropped = self.random_crop(image, depth_gt, shapeh, shapev, K_cropped, self.args.input_height, self.args.input_width)
            image, depth_gt, shapeh, shapev = self.train_preprocess(image, depth_gt, shapeh, shapev)
            sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'shapeh': shapeh, 'shapev': shapev, 'K': K_cropped}

        else:
            if self.mode == 'online_eval':
                data_path = self.args.data_path_eval
            else:
                data_path = self.args.data_path

            image_path = os.path.join(data_path, "./" + sample_path.split()[0])
            image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

            shapeh_path = os.path.join(self.args.angdata_path, 'angh', sample_path.split()[0].replace('/data', ''))
            shapev_path = os.path.join(self.args.angdata_path, 'angv', sample_path.split()[0].replace('/data', ''))

            shapeh = Image.open(shapeh_path)
            shapev = Image.open(shapev_path)
            shapeh = np.asarray(shapeh, dtype=np.float32) / 255.0 / 255.0
            shapev = np.asarray(shapev, dtype=np.float32) / 255.0 / 255.0
            shapeh = np.expand_dims(shapeh, axis=2)
            shapev = np.expand_dims(shapev, axis=2)

            if self.mode == 'online_eval':
                gt_path = self.args.gt_path_eval
                depth_path = os.path.join(gt_path, "./" + sample_path.split()[1])
                has_valid_depth = False
                gt_shape = np.zeros([1, 2])
                try:
                    depth_gt = Image.open(depth_path)
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    # print('Missing gt for {}'.format(image_path))

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    gt_shape[:] = depth_gt.shape[0:2]
                    depth_gt = depth_gt / 256.0

            if self.args.do_kb_crop is True:
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

                shapeh = shapeh[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                shapev = shapev[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                K_cropped[0, 2] = K_cropped[0, 2] - left_margin
                K_cropped[1, 2] = K_cropped[1, 2] - top_margin

            if self.mode == 'online_eval':
                sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth,
                          'gt_shape': gt_shape, 'shapeh': shapeh, 'shapev': shapev, 'K': K_cropped}
            else:
                sample = {'image': image, 'focal': focal, 'shapeh': shapeh, 'shapev': shapev, 'K': K_cropped}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, shapeh, shapev, K_cropped, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]

        shapeh = shapeh[y:y + height, x:x + width, :]
        shapev = shapev[y:y + height, x:x + width, :]

        K_cropped[0, 2] = K_cropped[0, 2] - x
        K_cropped[1, 2] = K_cropped[1, 2] - y
        return img, depth, shapeh, shapev, K_cropped

    def train_preprocess(self, image, depth_gt, shapeh, shapev):
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt, shapeh, shapev

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
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
        self.normalizeAng = transforms.Normalize(mean=[0.485], std=[0.229])

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)

        shapeh = self.to_tensor(sample['shapeh'])
        shapev = self.to_tensor(sample['shapev'])
        shapeh = self.normalizeAng(shapeh)
        shapev = self.normalizeAng(shapev)

        K = torch.from_numpy(sample['K']).float()

        if self.mode == 'test':
            return {'image': image, 'focal': focal, 'shapeh': shapeh, 'shapev': shapev, 'K': K}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth, 'focal': focal, 'shapeh': shapeh, 'shapev': shapev, 'K': K}
        else:
            has_valid_depth = sample['has_valid_depth']
            if 'gt_shape' in sample:
                return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth, 'gt_shape': sample['gt_shape'], 'shapeh': shapeh, 'shapev': shapev, 'K': K}
            else:
                return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth, 'shapeh': shapeh, 'shapev': shapev, 'K': K}

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


def get_intrinsic(calibpath, camind):
    cam2cam = read_calib_file(calibpath)
    K = np.eye(4)
    K[0:3, :] = cam2cam['P_rect_0{}'.format(camind)].reshape(3, 4)
    return K

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
