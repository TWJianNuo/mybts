import torch
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt

def tensor2rgb(tensor, viewind=0):
    tnp = tensor.detach().cpu().permute([0, 2, 3, 1]).contiguous()[viewind, :, :, :].numpy()
    tnp = tnp * np.array([[[0.229, 0.224, 0.225]]])
    tnp = tnp + np.array([[[0.485, 0.456, 0.406]]])
    tnp = tnp * 255
    tnp = np.clip(tnp, a_min=0, a_max=255).astype(np.uint8)
    return pil.fromarray(tnp)

def tensor2disp(tensor, vmax=0.18, percentile=None, viewind=0):
    cm = plt.get_cmap('magma')
    tnp = tensor[viewind, 0, :, :].detach().cpu().numpy()
    if percentile is not None:
        vmax = np.percentile(tnp, percentile)
    tnp = tnp / vmax
    tnp = (cm(tnp) * 255).astype(np.uint8)
    return pil.fromarray(tnp)