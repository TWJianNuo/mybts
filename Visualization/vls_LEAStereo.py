import torch
import glob
import PIL.Image as Image
import os
import numpy as np
from util import *
predroot = '/media/shengjie/disk1/Prediction/LEAStereo/kitti2015/disp_0'
for i in range(200):
    img = Image.open(os.path.join(predroot, "{}_10.png".format(str(i).zfill(6))))
    img = np.array(img).astype(np.float32)
    img = img / 256.0
    img = 0.54 * 7.215377e+02 / img
    depth = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()

    h, w = img.shape

    normoptimizer = SurfaceNormalOptimizer(height=h, width=w, batch_size=1)
    intirnsic = np.array(
        [[ 7.2154e+02,  0.0000e+00,  5.9356e+02,  4.4813e+01],
         [ 0.0000e+00,  7.2154e+02,  1.4985e+02,  1.5322e-01],
         [ 0.0000e+00,  0.0000e+00,  1.0000e+00,  2.7459e-03],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    intirnsic = torch.from_numpy(intirnsic).unsqueeze(0).float()
    ang = normoptimizer.depth2ang_log(depthMap=depth, intrinsic=intirnsic)

    minang = - np.pi / 3 * 2
    maxang = 2 * np.pi - np.pi / 3 * 2

    viewind = 0
    tensor2disp(ang[:, 0].unsqueeze(1) - minang, vmax=maxang, viewind=viewind).show()
    tensor2disp(ang[:, 1].unsqueeze(1) - minang, vmax=maxang, viewind=viewind).show()
    tensor2disp(1/depth, percentile=95).show()