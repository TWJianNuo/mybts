import os
import glob
import PIL.Image as pil
import numpy as np
import torch
from util import *
import copy
predroot = '/media/shengjie/disk1/visualization/btspred'
model_names = ['dsnet121_shapecons_we1', 'dsnet121_shapecons_we0', 'dsnet121_shapecons_w0', 'dsnet121_shapecons_we-1', 'dsnet121_shapecons_we-2']

tovls = list()
imgpaths = glob.glob(os.path.join(predroot, model_names[0], '*.png'))

vlsroot = os.path.join(predroot, 'vls')
os.makedirs(os.path.join(predroot, vlsroot), exist_ok=True)

for imgpath in imgpaths:
    figs = list()
    for k in range(len(model_names)):
        imgpathref = imgpath.replace(model_names[0], model_names[k])

        depthmap = np.array(pil.open(imgpathref)).astype(np.float) / 256.0
        depthmap = torch.from_numpy(depthmap).unsqueeze(0).unsqueeze(0)

        figs.append(np.array(tensor2disp(1 / depthmap, vmax=0.2, viewind=0)))
    figl = np.concatenate(figs[0:3], axis=0)
    figr = np.concatenate(figs[3:5] + [figs[2]], axis=0)
    fig = np.concatenate([figl, figr], axis=1)
    pil.fromarray(fig).save(os.path.join(vlsroot, imgpath.split('/')[-1]))



