import os,sys,inspect
import time
import argparse
import sys

import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

from bts import BtsModel

import torch

import matplotlib
import matplotlib.cm
from tqdm import tqdm

from Exp_trainwithShape.bts_dataloader_wshape import *
from util import *
from Exp_trainwithShape.bts_shape import BtsModelShape
from integrationModule import IntegrationConstrainFunction

parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')

# Dataset
parser.add_argument('--dataset', type=str, help='dataset to train on, kitti or nyu', default='kitti')
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--gt_path', type=str, help='path to the groundtruth data', required=True)
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)

parser.add_argument('--bts_size',                  type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)

model_names = ['dsnet121_shapecons_we0', 'dsnet121_shapecons_w0', 'dsnet121_shapecons_we-1', 'dsnet121_shapecons_we-2']
encoders = ['densenet121_bts', 'densenet121_bts', 'densenet121_bts', 'densenet121_bts']

args = parser.parse_args()

for k in range(len(model_names)):
    args.model_name = model_names[k]
    args.encoder = encoders[k]

    if args.gpu is None:
        checkpoint = torch.load(args.checkpoint_path)
    else:
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(args.checkpoint_path, map_location=loc)
    global_step = checkpoint['global_step']
    model.load_state_dict(checkpoint['model'])

    model = BtsModel(args)