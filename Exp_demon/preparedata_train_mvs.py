import os
import argparse

from joblib import Parallel, delayed
import numpy as np
import imageio
imageio.plugins.freeimage.download()
import h5py
from lz4.block import decompress
import imageio

from path import Path

path = os.path.join(os.path.dirname(os.path.abspath(__file__)))

def dump_example(dataset_name, args):
    print("Converting {:}.h5 ...".format(dataset_name))
    file = h5py.File(os.path.join(args.mvs_root, "{:}.h5".format(dataset_name)), "r")
    
    for (seq_idx, seq_name) in enumerate(file):
        if dataset_name == 'scenes11_train':
            scale = 0.4
        else:
            scale = 1

        if ((dataset_name == 'sun3d_train_1.6m_to_infm' and seq_idx == 7) or \
            (dataset_name == 'sun3d_train_0.4m_to_0.8m' and seq_idx == 15) or \
            (dataset_name == 'scenes11_train' and (seq_idx == 2758 or seq_idx == 4691 or seq_idx == 7023 or seq_idx == 11157 or seq_idx == 17168 or seq_idx == 19595))):
            continue        # Skip error files

        print("Processing sequence {:d}/{:d}".format(seq_idx, len(file)))
        dump_dir = os.path.join(args.dump_root, dataset_name + "_" + "{:05d}".format(seq_idx))
        if not os.path.isdir(dump_dir):
            os.mkdir(dump_dir)
        dump_dir = Path(dump_dir)
        sequence = file[seq_name]["frames"]["t0"]
        poses = []
        for (f_idx, f_name) in enumerate(sequence):
            frame = sequence[f_name]
            for dt_type in frame:
                dataset = frame[dt_type]
                img = dataset[...]
                if dt_type == "camera":
                    if f_idx == 0:
                        intrinsics = np.array([[img[0], 0, img[3]], [0, img[1], img[4]], [0, 0, 1]])
                    pose = np.array([[img[5],img[8],img[11],img[14]*scale], [img[6],img[9],img[12],img[15]*scale], [img[7],img[10],img[13],img[16]*scale]])
                    poses.append(pose.tolist())
                elif dt_type == "depth":
                    dimension = dataset.attrs["extents"]
                    depth = np.array(np.frombuffer(decompress(img.tobytes(), dimension[0] * dimension[1] * 2), dtype = np.float16)).astype(np.float32)
                    depth = depth.reshape(dimension[0], dimension[1])*scale

                    dump_depth_file = dump_dir/'{:04d}.npy'.format(f_idx)
                    np.save(dump_depth_file, depth)
                elif dt_type == "image":
                    img = imageio.imread(img.tobytes())
                    dump_img_file = dump_dir/'{:04d}.jpg'.format(f_idx)
                    # scipy.misc.imsave(dump_img_file, img)
                    imageio.imwrite(dump_img_file, img)

        dump_cam_file = dump_dir/'cam.txt'
        np.savetxt(dump_cam_file, intrinsics)
        poses_file = dump_dir/'poses.txt'
        np.savetxt(poses_file, np.array(poses).reshape(-1, 12), fmt='%.6e')

        if len(dump_dir.files('*.jpg')) < 2:
            dump_dir.rmtree()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')

    parser.add_argument('--mvs_root', type=str, help='model name', default='bts_nyu_v2')
    parser.add_argument('--dump_root', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts', default='densenet161_bts')
    args = parser.parse_args()

    SEQS = ['mvs_achteck_turm', 'mvs_breisach', 'mvs_citywall']
    for s in SEQS:
        dump_example(s, args)

