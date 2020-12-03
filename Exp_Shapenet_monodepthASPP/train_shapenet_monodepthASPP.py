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

import time
import argparse
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__))) # location of src

import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

import matplotlib
import matplotlib.cm
from tqdm import tqdm

from Exp_Shapenet_monodepthASPP.shapedataset import KittiShapeDataLoader, KittiShapeDataset
from util import *
from Exp_Shapenet_monodepthASPP.shapenet import ShapeNet
from torchvision import transforms
import torch.utils.data.distributed
import torch.distributed as dist

version_num = torch.__version__
version_num = ''.join(i for i in version_num if i.isdigit())
version_num = int(version_num.ljust(10, '0'))
if version_num > 1100000000:
    from torch.utils.tensorboard import SummaryWriter
else:
    from tensorboardX import SummaryWriter

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='bts_eigen_v2')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, desenet121_bts, densenet161_bts, resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts', default='densenet161_bts')
# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=True)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)

# Log and save
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=500)

# Training
parser.add_argument('--fix_first_conv_blocks',                 help='if set, will fix the first two conv blocks', action='store_true')
parser.add_argument('--fix_first_conv_block',                  help='if set, will fix the first conv block', action='store_true')
parser.add_argument('--bn_no_track_stats',                     help='if set, will not track running stats in batch norm layers', action='store_true')
parser.add_argument('--bts_size',                  type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument("--scheduler_step_size",       type=int,   help="step size of the scheduler", default=15)
parser.add_argument('--angw',                      type=float, default=0)
parser.add_argument('--vlossw',                    type=float, default=0.2)
parser.add_argument('--sclw',                      type=float, default=0)

# Preprocessing
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Multi-gpu training
parser.add_argument('--num_workers',               type=int,   help='number of threads to use for data loading', default=1)
parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                    'N processes per node, which has N GPUs. This is the '
                                                                    'fastest way to use PyTorch for either single node or '
                                                                    'multi node data parallel training', action='store_true',)
# Online eval
parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)
parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=500)

kbcroph = 352
kbcropw = 1216

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.mode == 'train' and not args.checkpoint_path:
    from bts import *

elif args.mode == 'train' and args.checkpoint_path:
    model_dir = os.path.dirname(args.checkpoint_path)
    model_name = os.path.basename(model_dir)
    import sys
    sys.path.append(model_dir)
    for key, val in vars(__import__(model_name)).items():
        if key.startswith('__') and key.endswith('__'):
            continue
        vars()[key] = val

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

def block_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__

def online_eval(model, normoptizer_eval, dataloader_eval, gpu, ngpus):
    eval_measures = torch.zeros(2).cuda(device=gpu)
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(args.gpu, non_blocking=True))
            K = torch.autograd.Variable(eval_sample_batched['K'].cuda(gpu, non_blocking=True))
            depth_gt = torch.autograd.Variable(eval_sample_batched['depth'].cuda(args.gpu, non_blocking=True))

            if 'depth' not in eval_sample_batched:
                continue

            pred_shape = model(image)
            pred_shape = F.interpolate(pred_shape, [kbcroph, kbcropw], mode='bilinear', align_corners=True)
            loss = normoptizer_eval.intergrationloss_ang_validation(ang=pred_shape, intrinsic=K, depthMap=depth_gt)

        eval_measures[0] += loss
        eval_measures[1] += 1

    if args.multiprocessing_distributed:
        group = dist.new_group([i for i in range(ngpus)])
        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

    if not args.multiprocessing_distributed or gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        L1Measure = float(eval_measures_cpu[0] / eval_measures_cpu[1])
        print('L1 Loss: %f, from %d eval samples' % (L1Measure, int(eval_measures_cpu[1])))
        return L1Measure
    return None

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # Create model
    model = ShapeNet()
    model.train()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("Total number of learning parameters: {}".format(num_params_update))

    if args.distributed:
        normoptizer = SurfaceNormalOptimizer(height=kbcroph, width=kbcropw, batch_size=int(args.batch_size / dist.get_world_size()), angw=args.angw, vlossw=args.vlossw, sclw=args.sclw)
        normoptizer_eval = SurfaceNormalOptimizer(height=kbcroph, width=kbcropw, batch_size=1, angw=args.angw, vlossw=args.vlossw, sclw=args.sclw)
        normoptizer.to(f'cuda:{args.gpu}')
        normoptizer_eval.to(f'cuda:{args.gpu}')
    else:
        normoptizer = SurfaceNormalOptimizer(height=kbcroph, width=kbcropw, batch_size=int(args.batch_size), angw=args.angw, vlossw=args.vlossw, sclw=args.sclw)
        normoptizer_eval = SurfaceNormalOptimizer(height=kbcroph, width=kbcropw, batch_size=1, angw=args.angw, vlossw=args.vlossw, sclw=args.sclw)
        normoptizer = normoptizer.cuda()
        normoptizer_eval = normoptizer_eval.cuda()

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model)
        model.cuda()

    if args.distributed:
        print("Model Initialized on GPU: {}".format(args.gpu))
    else:
        print("Model Initialized")

    global_step = 0
    best_measure_l1 = 1e3
    best_step = 0

    # Training parameters
    optimizer = torch.optim.Adam(list(model.module.encoder.parameters()) + list(model.module.decoder.parameters()), lr=args.learning_rate)
    model_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.scheduler_step_size, 0.1)
    model_just_loaded = False
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("Loading checkpoint '{}'".format(args.checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(args.checkpoint_path)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint_path, map_location=loc)
            global_step = checkpoint['global_step']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            try:
                best_measure_l1 = checkpoint['best_measure_l1']
                best_step = checkpoint['best_step']
            except KeyError:
                print("Could not load values for online evaluation")

            print("Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path, checkpoint['global_step']))
        else:
            print("No checkpoint found at '{}'".format(args.checkpoint_path))
        model_just_loaded = True

    if args.retrain:
        global_step = 0

    cudnn.benchmark = True

    dataloader = KittiShapeDataLoader(args, 'train')
    dataloader_eval = KittiShapeDataLoader(args, 'eval')

    # Logging
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(os.path.join(args.log_directory, args.model_name, 'summaries'), flush_secs=30)
        eval_summary_writer = SummaryWriter(os.path.join(args.log_directory, args.model_name, 'summaries_eval'), flush_secs=30)

    start_time = time.time()
    duration = 0

    var_sum = [var.sum() for var in model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)

    print("Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum/var_cnt))

    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch

    while epoch < args.num_epochs:
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)

        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()
            before_op_time = time.time()

            image = torch.autograd.Variable(sample_batched['image'].cuda(args.gpu, non_blocking=True))
            K = torch.autograd.Variable(sample_batched['K'].cuda(gpu, non_blocking=True))
            depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(args.gpu, non_blocking=True))

            pred_shape = model(image)
            pred_shape = F.interpolate(pred_shape, [kbcroph, kbcropw], mode='bilinear', align_corners=True)

            loss, hloss, vloss, inbdh, inbdv = normoptizer.intergrationloss_ang(ang=pred_shape, intrinsic=K, depthMap=depth_gt)
            loss.backward()
            optimizer.step()

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], loss: {:.12f}'.format(epoch, step, steps_per_epoch, global_step, loss))
                if np.isnan(loss.cpu().item()):
                    print('NaN in loss occurred. Aborting training.')
                    return -1

            duration += time.time() - before_op_time
            if global_step and global_step % args.log_freq == 0 and not model_just_loaded:
                var_sum = [var.sum() for var in model.parameters() if var.requires_grad]
                var_cnt = len(var_sum)
                var_sum = np.sum(var_sum)
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    print("{}".format(args.model_name))
                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | var sum: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(args.gpu, examples_per_sec, loss, var_sum.item(), var_sum.item()/var_cnt, time_sofar, training_time_left))

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    writer.add_scalar('loss/shapeloss', loss, global_step)
                    writer.add_scalar('loss/hloss', hloss, global_step)
                    writer.add_scalar('loss/vloss', vloss, global_step)
                    writer.add_scalar('numcounts/inbdh', inbdh, global_step)
                    writer.add_scalar('numcounts/inbdv', inbdv, global_step)
                    writer.add_scalar('var average', var_sum.item()/var_cnt, global_step)

                    minang = - np.pi / 3 * 2
                    maxang = 2 * np.pi - np.pi / 3 * 2

                    viewind = 0
                    depth_gtvls = torch.clone(depth_gt)
                    depth_gtvls[depth_gtvls == 0] = float("Inf")
                    fig_gt = tensor2disp(1 / depth_gtvls, vmax=0.15, viewind=viewind)
                    fig_rgb = tensor2rgb(sample_batched['image'], viewind=viewind).resize([kbcropw, kbcroph])

                    fig_angh = tensor2disp(pred_shape[:, 0].unsqueeze(1) - minang, vmax=maxang, viewind=viewind)
                    fig_angv = tensor2disp(pred_shape[:, 1].unsqueeze(1) - minang, vmax=maxang, viewind=viewind)

                    figoveiewu = np.concatenate([np.array(fig_rgb), np.array(fig_gt)], axis=1)
                    figoveiewd = np.concatenate([np.array(fig_angh), np.array(fig_angv)], axis=1)
                    figoveiew = np.concatenate([figoveiewu, figoveiewd], axis=0)

                    fignorm = normoptizer.ang2normal(ang=pred_shape, intrinsic=K)
                    fignorm = np.array(tensor2rgb((fignorm + 1) / 2, viewind=viewind, isnormed=False))

                    writer.add_image('oview', (torch.from_numpy(figoveiew).float() / 255).permute([2, 0, 1]), global_step)
                    writer.add_image('normal', (torch.from_numpy(fignorm).float() / 255).permute([2, 0, 1]), global_step)
                    if version_num > 1100000000:
                        writer.flush()

            if not args.do_online_eval and global_step and global_step % args.save_freq == 0:
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    checkpoint = {'global_step': global_step,
                                  'model': model.state_dict(),
                                  'optimizer': optimizer.state_dict()}
                    torch.save(checkpoint, args.log_directory + '/' + args.model_name + '/model-{}'.format(global_step))

            if args.do_online_eval and global_step and global_step % args.eval_freq == 0 and not model_just_loaded and epoch>=2:
                time.sleep(0.1)
                model.eval()
                eval_measure = online_eval(model, normoptizer_eval, dataloader_eval, gpu, ngpus_per_node)
                if eval_measure is not None:
                    if eval_measure < best_measure_l1:
                        old_best_measure_l1 = best_measure_l1
                        old_best_step = best_step
                        best_measure_l1 = eval_measure
                        best_step = global_step
                        is_best = True
                    else:
                        is_best = False
                    print("Best L1 Performance: %f, at step %d" % (best_measure_l1, best_step))
                    eval_summary_writer.add_scalar('L1Measure', eval_measure, int(global_step))
                    if is_best:
                        old_best_name = 'model-{}-best_{}_{:.5f}'.format(old_best_step, 'L1Measure', old_best_measure_l1)
                        model_path = os.path.join(args.log_directory, args.model_name, old_best_name)
                        if os.path.exists(model_path):
                            command = 'rm {}'.format(model_path)
                            os.system(command)
                        model_save_name = 'model-{}-best_{}_{:.5f}'.format(best_step, 'L1Measure', best_measure_l1)
                        print('New best for {}. Saving model: {}'.format('L1Measure', model_save_name))
                        checkpoint = {'global_step': global_step,
                                      'model': model.state_dict(),
                                      'optimizer': optimizer.state_dict(),
                                      'best_measure_l1': best_measure_l1,
                                      'best_step': best_step
                                      }
                        torch.save(checkpoint, os.path.join(args.log_directory, args.model_name, model_save_name))
                    if version_num > 1100000000:
                        eval_summary_writer.flush()
                model.train()
                block_print()
                enable_print()

            model_just_loaded = False
            global_step += 1

        epoch += 1
        model_lr_scheduler.step()

def main():
    if args.mode != 'train':
        print('bts_main.py is only for training. Use bts_test.py instead.')
        return -1

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print("This will evaluate the model every eval_freq {} steps and save best models for individual eval metrics."
              .format(args.eval_freq))

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()
