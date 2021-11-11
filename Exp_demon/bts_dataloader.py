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
import copy
import random
import numpy as np
import transforms as T

from PIL import Image
from glob import glob

from distributed_sampler_no_evenly_divisible import *
from fvcore.transforms.transform import CropTransform

import torch.utils.data.distributed
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transforms.augmentation import _transform_to_aug

class BtsDataLoader(object):
    def __init__(self, args, is_test=True):
        self.demon = DeMoN(args.demon_path, args, is_test)
        if not is_test:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.demon) if args.distributed else None
            self.data = DataLoader(self.demon, args.batch_size, shuffle=(self.train_sampler is None), num_workers=args.num_threads, pin_memory=True, sampler=self.train_sampler, drop_last=True)
        else:
            self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.demon, shuffle=False) if args.distributed else None
            self.data = DataLoader(self.demon, 1, shuffle=False, num_workers=1, pin_memory=True, sampler=self.eval_sampler, drop_last=False)


class DeMoN(Dataset):
    def __init__(self, root, args, is_test=True):
        self.args = args
        self.root = root
        self.is_test = is_test
        self.color_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.input_height = self.args.input_height
        self.input_width = self.args.input_width

        self.MAXFLOW = 1e10

        if self.is_test:
            self.split = 'test'
        else:
            self.split = 'train'

        catcounts = {'sun3d': 0,
                     'rgbd': 0,
                     'scenes11': 0,
                     'mvs': 0
                     }

        self.entries = list()
        for seq in glob(os.path.join(self.args.demon_path, self.split, '*/')):
            jpgpaths = glob(os.path.join(seq, '*.jpg'))
            for idx in range(len(jpgpaths)):
                self.entries.append("{} {}".format(seq.split('/')[-2], str(idx)))
            catcounts[seq.split('/')[-2].split('_')[0]] += len(jpgpaths)

        if not is_test:
            self.crop_flip = T.AugmentationList([
                T.RandomCrop(crop_type='absolute', crop_size=[self.args.input_height, self.args.input_width]),
                T.RandomFlip(prob=0.5, horizontal=True)])

    def __getitem__(self, idx):
        entry = self.entries[idx]
        seqname, jpgidx = entry.split(' ')
        jpgidx = int(jpgidx)

        intrinsic_path = os.path.join(self.root, self.split, seqname, 'cam.txt')
        K = np.eye(4)
        K[0:3, 0:3] = copy.deepcopy(np.genfromtxt(intrinsic_path).astype(np.float32).reshape((3, 3)))

        # Read RGB
        image = Image.open(os.path.join(self.root, self.split, seqname, "{}.jpg".format(str(jpgidx).zfill(4))))

        # Read Depth
        depth_gt = np.load(os.path.join(self.root, self.split, seqname, "{}.npy".format(str(jpgidx).zfill(4))))
        depth_gt[np.isnan(depth_gt)] = 0
        depth_gt[np.isinf(depth_gt)] = 0

        datasetname = seqname.split('_')[0]
        if datasetname == 'scenes11':
            depth_gt = depth_gt / 5

        # Read Size
        size = np.array(image.size)

        # Convert to numpy
        image, depth_gt = self.cvt_np(image=image, depth_gt=depth_gt)

        # Augmentation in Training
        if not self.is_test:
            # Random Rotation
            if self.args.do_random_rotate:
                image, depth_gt = self.do_random_rotate(image=image, depth_gt=depth_gt)

            # Random Crop and Flip
            image, depth_gt, K = self.do_random_crop_flip(image=image, depth_gt=depth_gt, K=K)
            image = self.do_random_coloraug(image=image)

        image, depth_gt, K = self.to_tensor(image=image, depth_gt=depth_gt, K=K)
        image = self.color_normalize(image)
        sample = {'image': image, 'depth': depth_gt, 'K': K, 'size': size, 'entry':[entry]}
        return sample

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

    def do_random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

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
        depth_gt = torch.from_numpy(np.copy(depth_gt)).unsqueeze(0).contiguous().float()
        K = torch.from_numpy(K).float()
        return image, depth_gt, K

    def __len__(self):
        return len(self.entries)