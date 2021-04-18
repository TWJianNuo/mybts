import os
import matplotlib.pyplot as plt
import PIL.Image as pil
import numpy as np
files = '/home/shengjie/Documents/Project_SemanticDepth/splits/eigen_full/test_files.txt'
kittiroot = '/home/shengjie/Documents/Data/Kitti/kitti_raw/kitti_data'
raw_lidar_root = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Data/kitti/unprocessed_raw_lidar_mapped_depth'
filter_lidar_root = '/home/shengjie/Documents/Data/Kitti/filtered_lidar'
with open(files) as f:
    content = f.readlines()

for k, entry in enumerate(content):
    if k != 324:
        continue
    seq, idx, dir = entry.split(' ')

    rgb = os.path.join(kittiroot, seq, 'image_02', 'data', idx.zfill(10) + '.png')
    rgb = pil.open(rgb)

    depthamp1 = os.path.join(raw_lidar_root, seq, 'image_02', idx.zfill(10) + '.png')
    depthamp1 = pil.open(depthamp1)
    depthamp1 = np.array(depthamp1).astype(np.float32) / 256.0

    depthamp2 = os.path.join(filter_lidar_root, seq, 'image_02', idx.zfill(10) + '.png')
    depthamp2 = pil.open(depthamp2)
    depthamp2 = np.array(depthamp2).astype(np.float32) / 256.0

    w, h = rgb.size
    xx, yy = np.meshgrid(range(w), range(h), indexing='xy')

    cm = plt.get_cmap('magma')
    vmax = 0.15

    selector = depthamp1 > 0
    xxs = xx[selector]
    yys = yy[selector]
    dp = 1 / depthamp1[selector]
    dp = dp / vmax
    dp = cm(dp)

    plt.figure()
    plt.imshow(rgb)
    plt.scatter(xxs, yys, 1, dp[:, 0:3])

    selector = depthamp2 > 0
    xxs = xx[selector]
    yys = yy[selector]
    dp = 1 / depthamp2[selector]
    dp = dp / vmax
    dp = cm(dp)

    plt.figure()
    plt.imshow(rgb)
    plt.scatter(xxs, yys, 1, dp[:, 0:3])
    a = 1


a = 1