import torch
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import torch.nn as nn

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
        self.init_scalIndex()
        self.init_patchIntPath()

    def init_scalIndex(self):
        self.xxdict = dict()
        self.yydict = dict()
        self.ccdict = dict()
        for i in range(1, 4):
            cuh = int(self.height / (2 ** i))
            cuw = int(self.width / (2 ** i))

            cx = int(0 + 2 ** i / 2)
            cy = int(0 + 2 ** i / 2)

            cxx, cyy = np.meshgrid(range(cuw), range(cuh), indexing='xy')

            cxx = cxx * (2**i) + cx
            cyy = cyy * (2**i) + cy

            cxxt = nn.Parameter(torch.from_numpy(np.copy(cxx)).unsqueeze(0).repeat([self.batch_size, 1, 1]).long(), requires_grad=False)
            cyyt = nn.Parameter(torch.from_numpy(np.copy(cyy)).unsqueeze(0).repeat([self.batch_size, 1, 1]).long(), requires_grad=False)
            ccct = nn.Parameter(torch.zeros([self.batch_size, cuh, cuw], dtype=torch.long), requires_grad=False)

            self.xxdict["scale_{}".format(i)] = cxxt
            self.yydict["scale_{}".format(i)] = cyyt
            self.ccdict["scale_{}".format(i)] = ccct

    def init_patchIntPath(self):
        self.patchIntPath = dict()
        for i in range(1, 4):
            intpath = list()
            for m in range(0, int(2 ** i)):
                for n in range(0, int(2 ** i)):
                    if m == 0 and n == 0:
                        continue
                    if np.mod(m, 2) == 1:
                        mm = int(-(m+1)/2)
                    else:
                        mm = int(m/2)
                    if np.mod(n, 2) == 1:
                        nn = int(-(n+1)/2)
                    else:
                        nn = int(n/2)

                    if nn < 0:
                        logx1 = nn
                        logy1 = mm
                        ch1 = 0
                        depthx1 = nn + 1
                        depthy1 = mm
                        sign1 = -1
                    elif nn > 0:
                        logx1 = nn - 1
                        logy1 = mm
                        ch1 = 0
                        depthx1 = nn - 1
                        depthy1 = mm
                        sign1 = 1

                    if mm > 0:
                        logx2 = nn
                        logy2 = mm - 1
                        ch2 = 1
                        depthx2 = nn
                        depthy2 = mm - 1
                        sign2 = 1
                    elif mm < 0:
                        logx2 = nn
                        logy2 = mm
                        ch2 = 1
                        depthx2 = nn
                        depthy2 = mm + 1
                        sign2 = -1

                    if nn == 0:
                        logx1 = logx2
                        logy1 = logy2
                        ch1 = ch2
                        depthx1 = depthx2
                        depthy1 = depthy2
                        sign1 = sign2

                    if mm == 0:
                        logx2 = logx1
                        logy2 = logy1
                        ch2 = ch1
                        depthx2 = depthx1
                        depthy2 = depthy1
                        sign2 = sign1

                    intpath.append((logx1, logy1, ch1, depthx1, depthy1, sign1, logx2, logy2, ch2, depthx2, depthy2, sign2, mm, nn))
                    self.patchIntPath['scale_{}'.format(i)] = intpath

    def patchIntegration(self, depthmaplow, ang, intrinsic, scale):
        log = self.ang2log(intrinsic=intrinsic, ang=ang)
        depthmaplowl = torch.log(depthmaplow.squeeze(1))
        _, lh, lw = depthmaplowl.shape
        assert (lh == self.height / (2 ** scale)) and (lw == self.width / (2 ** scale)), print("Resolution and scale are not cooresponded")

        xx = self.xxdict["scale_{}".format(scale)]
        yy = self.yydict["scale_{}".format(scale)]
        cc = self.ccdict["scale_{}".format(scale)]
        intpaths = self.patchIntPath['scale_{}'.format(scale)]
        depthmapr = torch.zeros([self.batch_size, self.height, self.width], device="cuda")
        depthmapr[:, yy[0], xx[0]] = depthmaplowl

        for path in intpaths:
            logx1, logy1, ch1, depthx1, depthy1, sign1, logx2, logy2, ch2, depthx2, depthy2, sign2, mm, nn = path
            tmpgradnode = depthmapr.clone()
            depthmapr[:, yy + mm, xx + nn] = (tmpgradnode[:, yy + depthy1, xx + depthx1] + sign1 * log[:, cc + ch1, yy + logy1, xx + logx1] +
                                              tmpgradnode[:, yy + depthy2, xx + depthx2] + sign2 * log[:, cc + ch2, yy + logy2, xx + logx2]) / 2

        depthmapr = torch.exp(depthmapr).unsqueeze(1)
        # tensor2disp(depthmaplow, vmax=40, ind=0).show()
        # tensor2disp(depthmapr, vmax=40, ind=0).show()
        return depthmapr

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