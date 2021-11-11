import glob, os, random
import numpy as np
import matplotlib.pyplot as plt

import torch
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

def tensor2disp(tensor, vmax=0.18, percentile=None, viewind=0):
    cm = plt.get_cmap('magma')
    tnp = tensor[viewind, 0, :, :].detach().cpu().numpy()
    if percentile is not None:
        vmax = np.percentile(tnp, percentile)
    tnp = tnp / vmax
    tnp = (cm(tnp) * 255).astype(np.uint8)
    return pil.fromarray(tnp[:, :, 0:3])

if __name__ == '__main__':
    entries = list()
    demon_path = '/media/shengjie/scratch2/DeMoN/DeepSFM/dataset'

    sun3dentries = list()
    scenes11entries = list()
    rgbdentries = list()
    for seq in glob.glob(os.path.join(demon_path, 'train', '*/')):
        if 'scenes11' in seq:
            # if 'sun3d' not in seq:
            #     continue
            jpgpaths = glob.glob(os.path.join(seq, '*.jpg'))
            for idx in range(len(jpgpaths)):
                scenes11entries.append("{} {}".format(seq.split('/')[-2], str(idx)))
        elif 'sun3d' in seq:
            jpgpaths = glob.glob(os.path.join(seq, '*.jpg'))
            for idx in range(len(jpgpaths)):
                sun3dentries.append("{} {}".format(seq.split('/')[-2], str(idx)))
        elif 'rgbd' in seq:
            jpgpaths = glob.glob(os.path.join(seq, '*.jpg'))
            for idx in range(len(jpgpaths)):
                rgbdentries.append("{} {}".format(seq.split('/')[-2], str(idx)))

    samplenum = 500

    random.shuffle(scenes11entries)
    scenes11entries = scenes11entries[0:samplenum]
    random.shuffle(sun3dentries)
    sun3dentries = sun3dentries[0:samplenum]
    random.shuffle(rgbdentries)
    rgbdentries = rgbdentries[0:samplenum]

    # scenes11entries.append('scenes11_train_19547 0')
    # scenes11entries = scenes11entries[::-1]
    #
    # scene11list = list()
    # for entry in scenes11entries:
    #     seqname, jpgidx = entry.split(' ')
    #     depth_gt = np.load(os.path.join(demon_path, 'train', seqname, "{}.npy".format(str(jpgidx).zfill(4))))
    #     depth_gt[np.isnan(depth_gt)] = 0
    #     depth_gt[np.isinf(depth_gt)] = 0
    #     selector = (depth_gt > 0) * (depth_gt < 50)
    #
    #     # tensor2disp(torch.from_numpy(selector).unsqueeze(0).unsqueeze(0)).show()
    #
    #     scene11list.append(depth_gt[selector])
    #
    # scene11list = np.concatenate(scene11list)
    #
    # plt.figure()
    # plt.hist(scene11list, density=True, bins=50)  # density=False would make counts
    # plt.show()

    # sun3dlist = list()
    # for entry in sun3dentries:
    #     seqname, jpgidx = entry.split(' ')
    #     depth_gt = np.load(os.path.join(demon_path, 'train', seqname, "{}.npy".format(str(jpgidx).zfill(4))))
    #     depth_gt[np.isnan(depth_gt)] = 0
    #     depth_gt[np.isinf(depth_gt)] = 0
    #     selector = (depth_gt > 0)
    #
    #     sun3dlist.append(depth_gt[selector])
    #
    # sun3dlist = np.concatenate(sun3dlist)
    #
    # plt.figure()
    # plt.hist(sun3dlist, density=True, bins=50)  # density=False would make counts
    # plt.show()

    rgbdlist = list()
    for entry in rgbdentries:
        seqname, jpgidx = entry.split(' ')
        depth_gt = np.load(os.path.join(demon_path, 'train', seqname, "{}.npy".format(str(jpgidx).zfill(4))))
        depth_gt[np.isnan(depth_gt)] = 0
        depth_gt[np.isinf(depth_gt)] = 0
        selector = (depth_gt > 0)

        rgbdlist.append(depth_gt[selector])

    rgbdlist = np.concatenate(rgbdlist)

    plt.figure()
    plt.hist(rgbdlist, density=True, bins=50)  # density=False would make counts
    plt.show()