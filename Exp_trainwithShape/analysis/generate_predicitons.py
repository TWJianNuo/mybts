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
import glob


parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')

# Dataset
parser.add_argument('--dataset',                type=str, help='dataset to train on, kitti or nyu', default='kitti')
parser.add_argument('--data_path',              type=str, help='path to the data', required=True)
parser.add_argument('--gt_path',                type=str, help='path to the groundtruth data', required=True)
parser.add_argument('--angdata_path',           type=str,   default=None)
parser.add_argument('--inslabel_path',          type=str,   default=None)
parser.add_argument('--filenames_file_eval',    type=str, help='path to the filenames text file', required=True)
parser.add_argument('--input_height',           type=int, help='input height', default=480)
parser.add_argument('--input_width',            type=int, help='input width', default=640)
parser.add_argument('--weight_path',            type=str)
parser.add_argument('--limitvalnum',            type=int, default=None)
parser.add_argument('--outputfold',             type=str)
parser.add_argument('--min_depth_eval',         type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',         type=float, help='maximum depth for evaluation', default=80)

parser.add_argument('--bts_size',               type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--max_depth',              type=float, help='maximum depth in estimation', default=80)
parser.add_argument('--batch_size',             type=int,  default=1)


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]


model_names = ['dsnet121_shapecons_we-2', 'dsnet121_shapecons_w0', 'dsnet121_shapecons_we0',    'dsnet121_shapecons_we-1', 'dsnet121_shapecons_we1']
encoders =    ['densenet121_bts',          'densenet121_bts',          'densenet121_bts',          'densenet121_bts', 'densenet121_bts']
selected_metric = 'd1'

args = parser.parse_args()
args.distributed = False
args.do_kb_crop = True

args.num_threads = 12
args.dataset = 'kitti'
dataloader_eval = BtsDataLoader(args, 'online_eval')

for k in range(len(model_names)):
    args.model_name = model_names[k]
    args.encoder = encoders[k]

    model = BtsModel(args)
    checkpointpaths = glob.glob(os.path.join(args.weight_path, args.model_name, 'model-*'))
    for checkpointpath in checkpointpaths:
        if selected_metric in checkpointpath.split('/')[-1]:
            checkpoint = torch.load(checkpointpath)
            translatedcheckpoint = dict()
            for key in checkpoint['model'].keys():
                newk = key[7::]
                translatedcheckpoint[newk] = checkpoint['model'][key]
            model.load_state_dict(translatedcheckpoint)

    svfoldpath = os.path.join(args.outputfold, model_names[k])
    os.makedirs(svfoldpath, exist_ok=True)

    model.eval().cuda()

    eval_measures = torch.zeros(10).cuda()
    for count, eval_sample_batched in enumerate(dataloader_eval.data):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda())
            gt_depth = torch.autograd.Variable(eval_sample_batched['depth'].cuda())
            focal = torch.autograd.Variable(eval_sample_batched['focal'].cuda())
            gt_shape = eval_sample_batched['gtshape']
            _, _, _, _, pred_depth = model(image, focal)

            fname = dataloader_eval.testing_samples.filenames[count]
            fname = "{}_{}".format(fname.split(' ')[0].split('/')[1],fname.split(' ')[0].split('/')[-1])

            prednp = (pred_depth[0,0,:,:].cpu().numpy() * 256.0).astype(np.uint16)
            prednp = pil.fromarray(prednp).save(os.path.join(svfoldpath, fname))

            gt_depth = gt_depth.cpu().numpy().squeeze()
            pred_depth = pred_depth.cpu().numpy().squeeze()

            pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
            pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
            pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
            pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

            gt_width, gt_height = gt_shape.numpy()[0, :]
            eval_mask = np.zeros(valid_mask.shape)

            top_margin = int(gt_height - 352)
            left_margin = int((gt_width - 1216) / 2)

            eval_mask[int(0.40810811 * gt_height) - top_margin:int(0.99189189 * gt_height) - top_margin,
            int(0.03594771 * gt_width) - left_margin:int(0.96405229 * gt_width) - left_margin] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

            measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

            eval_measures[:9] += torch.tensor(measures).cuda()
            eval_measures[9] += 1

    # print(eval_measures)
    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[9].item()
    eval_measures_cpu /= cnt
    print('Computing errors for {} eval samples of model {}'.format(int(cnt), model_names[k]))
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
    for i in range(8):
        print('{:7.3f}, '.format(eval_measures_cpu[i]), end='')
    print('{:7.3f}'.format(eval_measures_cpu[8]))

