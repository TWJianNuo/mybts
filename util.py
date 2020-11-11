import torch
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

def tensor2rgb(tensor, viewind=0):
    tnp = tensor.detach().cpu().permute([0, 2, 3, 1]).contiguous()[viewind, :, :, :].numpy()
    tnp = tnp * np.array([[[0.229, 0.224, 0.225]]])
    tnp = tnp + np.array([[[0.485, 0.456, 0.406]]])
    tnp = tnp * 255
    tnp = np.clip(tnp, a_min=0, a_max=255).astype(np.uint8)
    return pil.fromarray(tnp)

def tensor2rgb_normal(tensor, viewind=0):
    tnp = tensor.detach().cpu().permute([0, 2, 3, 1]).contiguous()[viewind, :, :, :].numpy()
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
    return pil.fromarray(tnp[:, :, 0:3])


class SurfaceNormalOptimizer(nn.Module):
    def __init__(self, height, width, batch_size):
        super(SurfaceNormalOptimizer, self).__init__()
        self.height = height
        self.width = width
        self.batch_size = batch_size

        xx, yy = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.xx = nn.Parameter(torch.from_numpy(np.copy(xx)).unsqueeze(0).repeat([self.batch_size, 1, 1]).float(), requires_grad=False)
        self.yy = nn.Parameter(torch.from_numpy(np.copy(yy)).unsqueeze(0).repeat([self.batch_size, 1, 1]).float(), requires_grad=False)

        xx = xx.flatten().astype(np.float32)
        yy = yy.flatten().astype(np.float32)
        pix_coords = np.expand_dims(np.stack([xx, yy, np.ones(self.width * self.height).astype(np.float32)], axis=1), axis=0).repeat(self.batch_size, axis=0)
        self.pix_coords = nn.Parameter(torch.from_numpy(pix_coords).permute(0, 2, 1), requires_grad=False)
        self.ones = nn.Parameter(torch.ones([self.batch_size, 1, self.height * self.width]), requires_grad=False)
        self.init_gradconv()

    def init_gradconv(self):
        weightsx = torch.Tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        weightsx = weightsx / 4 / 2

        weightsy = torch.Tensor([[-1., -2., -1.],
                                 [0., 0., 0.],
                                 [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        weightsy = weightsy / 4 / 2
        self.diffx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.diffy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.diffx.weight = nn.Parameter(weightsx, requires_grad=False)
        self.diffy.weight = nn.Parameter(weightsy, requires_grad=False)

        weightsx = torch.Tensor([[0., 0., 0.],
                                [0., -1., 1.],
                                [0., 0., 0.]]).unsqueeze(0).unsqueeze(0)
        weightsx = weightsx

        weightsy = torch.Tensor([[0., 0., 0.],
                                 [0., -1., 0.],
                                 [0., 1., 0.]]).unsqueeze(0).unsqueeze(0)
        weightsy = weightsy
        self.diffx_sharp = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.diffy_sharp = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.diffx_sharp.weight = nn.Parameter(weightsx, requires_grad=False)
        self.diffy_sharp.weight = nn.Parameter(weightsy, requires_grad=False)

    def ang2log(self, intrinsic, ang):
        protectmin = 1e-6

        angh = ang[:, 0, :, :]
        angv = ang[:, 1, :, :]

        fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        bx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        by = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])

        a1 = ((self.yy - by) / fy)**2 + 1
        b1 = -(self.xx - bx) / fx

        a2 = ((self.yy - by) / fy)**2 + 1
        b2 = -(self.xx + 1 - bx) / fx

        a3 = torch.sin(angh)
        b3 = -torch.cos(angh)

        u1 = ((self.xx - bx) / fx)**2 + 1
        v1 = -(self.yy - by) / fy

        u2 = ((self.xx - bx) / fx)**2 + 1
        v2 = -(self.yy + 1 - by) / fy

        u3 = torch.sin(angv)
        v3 = -torch.cos(angv)

        logh = torch.log(torch.clamp(torch.abs(a3 * b1 - a1 * b3), min=protectmin)) - torch.log(torch.clamp(torch.abs(a3 * b2 - a2 * b3), min=protectmin))
        logh = torch.clamp(logh, min=-10, max=10)

        logv = torch.log(torch.clamp(torch.abs(u3 * v1 - u1 * v3), min=protectmin)) - torch.log(torch.clamp(torch.abs(u3 * v2 - u2 * v3), min=protectmin))
        logv = torch.clamp(logv, min=-10, max=10)

        return torch.stack([logh, logv], dim=1)

    def ang2normal(self, ang, intrinsic):
        fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        bx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        by = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])

        angh = ang[:, 0, :, :]
        angv = ang[:, 1, :, :]

        normx = torch.stack([torch.cos(angh), (self.yy - by) / fy * torch.sin(angh), torch.sin(angh)], dim=1)
        normy = torch.stack([(self.xx - bx) / fx * torch.sin(angv), torch.cos(angv), torch.sin(angv)], dim=1)

        surfacenormal = torch.cross(normx, normy, dim=1)
        surfacenormal = F.normalize(surfacenormal, dim=1)
        surfacenormal = torch.clamp(surfacenormal, min=-1+1e-6, max=1-1e-6)

        return surfacenormal

    def depth2norm(self, depthMap, intrinsic, issharp=True):
        depthMaps = depthMap.squeeze(1)
        fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        bx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        by = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])

        if issharp:
            depthMap_gradx = self.diffx_sharp(depthMap).squeeze(1)
            depthMap_grady = self.diffy_sharp(depthMap).squeeze(1)
        else:
            depthMap_gradx = self.diffx(depthMap).squeeze(1)
            depthMap_grady = self.diffy(depthMap).squeeze(1)

        vx1 = depthMaps / fx + (self.xx - bx) / fx * depthMap_gradx
        vx2 = (self.yy - by) / fy * depthMap_gradx
        vx3 = depthMap_gradx

        vy1 = (self.xx - bx) / fx * depthMap_grady
        vy2 = depthMaps / fy + (self.yy - by) / fy * depthMap_grady
        vy3 = depthMap_grady

        vx = torch.stack([vx1, vx2, vx3], dim=1)
        vy = torch.stack([vy1, vy2, vy3], dim=1)

        surfnorm = torch.cross(vx, vy, dim=1)
        surfnorm = F.normalize(surfnorm, dim=1)
        surfnorm = torch.clamp(surfnorm, min=-1+1e-6, max=1-1e-6)

        return surfnorm