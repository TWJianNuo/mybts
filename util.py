import torch
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

def tensor2rgb(tensor, viewind=0, isnormed=True):
    tnp = tensor.detach().cpu().permute([0, 2, 3, 1]).contiguous()[viewind, :, :, :].numpy()
    if isnormed:
        tnp = tnp * np.array([[[0.229, 0.224, 0.225]]])
        tnp = tnp + np.array([[[0.485, 0.456, 0.406]]])
        tnp = tnp * 255
        tnp = np.clip(tnp, a_min=0, a_max=255).astype(np.uint8)
    else:
        tnp = (tnp * 255).astype(np.uint8)
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

def tensor2grad(gradtensor, percentile=95, pos_bar=0, neg_bar=0, viewind=0):
    cm = plt.get_cmap('bwr')
    gradnumpy = gradtensor.detach().cpu().numpy()[viewind, 0, :, :]

    selector_pos = gradnumpy > 0
    if np.sum(selector_pos) > 1:
        if pos_bar <= 0:
            pos_bar = np.percentile(gradnumpy[selector_pos], percentile)
        gradnumpy[selector_pos] = gradnumpy[selector_pos] / pos_bar / 2

    selector_neg = gradnumpy < 0
    if np.sum(selector_neg) > 1:
        if neg_bar >= 0:
            neg_bar = -np.percentile(-gradnumpy[selector_neg], percentile)
        gradnumpy[selector_neg] = -gradnumpy[selector_neg] / neg_bar / 2

    disp_grad_numpy = gradnumpy + 0.5
    colorMap = cm(disp_grad_numpy)[:,:,0:3]
    return pil.fromarray((colorMap * 255).astype(np.uint8))

def tensor2disp_circ(tensor, vmax=0.18, percentile=None, viewind=0):
    cm = plt.get_cmap('hsv')
    tnp = tensor[viewind, 0, :, :].detach().cpu().numpy()
    selector = tnp == 0
    selector = np.stack([selector, selector, selector], axis=2)

    if percentile is not None:
        vmax = np.percentile(tnp, percentile)
    tnp = tnp / vmax
    tnp = np.mod(tnp, 1)
    tnp = (cm(tnp) * 255).astype(np.uint8)
    tnp = tnp[:, :, 0:3]

    tnp[selector] = 0
    return pil.fromarray(tnp)

class SurfaceNormalOptimizer(nn.Module):
    def __init__(self, height, width, batch_size, angw=1e-6, vlossw=0.2, sclw=0):
        super(SurfaceNormalOptimizer, self).__init__()
        # intrinsic: (batch_size, 4, 4)
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.sclw = sclw
        self.angw = angw
        self.vlossw = vlossw

        xx, yy = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.xx = nn.Parameter(torch.from_numpy(np.copy(xx)).unsqueeze(0).repeat([self.batch_size, 1, 1]).float(), requires_grad=False)
        self.yy = nn.Parameter(torch.from_numpy(np.copy(yy)).unsqueeze(0).repeat([self.batch_size, 1, 1]).float(), requires_grad=False)

        xx = xx.flatten().astype(np.float32)
        yy = yy.flatten().astype(np.float32)
        pix_coords = np.expand_dims(np.stack([xx, yy, np.ones(self.width * self.height).astype(np.float32)], axis=1), axis=0).repeat(self.batch_size, axis=0)
        self.pix_coords = nn.Parameter(torch.from_numpy(pix_coords).permute(0, 2, 1), requires_grad=False)
        self.ones = nn.Parameter(torch.ones([self.batch_size, 1, self.height * self.width]), requires_grad=False)
        self.init_gradconv()
        self.init_integration_kernel()

    def init_integration_kernel(self):
        inth = torch.Tensor(
            [[0, 1, 0, 0, 0],
             [0, 1, 1, 0, 0],
             [1, 1, 1, 0, 0],
             [1, 1, 1, 1, 0],
            ]
        )
        gth = torch.Tensor(
            [[0, -1, 1, 0, 0],
             [0, -1, 0, 1, 0],
             [-1, 0, 0, 1, 0],
             [-1, 0, 0, 0, 1],
            ]
        )
        idh = torch.Tensor(
            [[0, 1, 1, 0, 0],
             [0, 1, 0, 1, 0],
             [1, 0, 0, 1, 0],
             [1, 0, 0, 0, 1],
            ]
        )
        self.inth = torch.nn.Conv2d(1, 4, [1, 5], padding=[0, 2], bias=False)
        self.inth.weight = torch.nn.Parameter(inth.unsqueeze(1).unsqueeze(1).float(), requires_grad=False)
        self.gth = torch.nn.Conv2d(1, 4, [1, 5], padding=[0, 2], bias=False)
        self.gth.weight = torch.nn.Parameter(gth.unsqueeze(1).unsqueeze(1).float(), requires_grad=False)
        self.idh = torch.nn.Conv2d(1, 4, [1, 5], padding=[0, 2], bias=False)
        self.idh.weight = torch.nn.Parameter(idh.unsqueeze(1).unsqueeze(1).float(), requires_grad=False)


        intv = torch.Tensor(
            [[1, 1, 1, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 0],
            ]
        )
        gtv = torch.Tensor(
            [[-1, 0, 0, 1, 0, 0, 0, 0, 0],
             [-1, 0, 0, 0, 1, 0, 0, 0, 0],
             [-1, 0, 0, 0, 0, 1, 0, 0, 0],
             [-1, 0, 0, 0, 0, 0, 1, 0, 0],
             [-1, 0, 0, 0, 0, 0, 0, 1, 0],
             [-1, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        idv = torch.Tensor(
            [[1, 0, 0, 1, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 1, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 1, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 1, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        self.intv = torch.nn.Conv2d(1, 6, [9, 1], padding=[4, 0], bias=False)
        self.intv.weight = torch.nn.Parameter(intv.unsqueeze(1).unsqueeze(3).float(), requires_grad=False)
        self.gtv = torch.nn.Conv2d(1, 6, [9, 1], padding=[4, 0], bias=False)
        self.gtv.weight = torch.nn.Parameter(gtv.unsqueeze(1).unsqueeze(3).float(), requires_grad=False)
        self.idv = torch.nn.Conv2d(1, 6, [9, 1], padding=[4, 0], bias=False)
        self.idv.weight = torch.nn.Parameter(idv.unsqueeze(1).unsqueeze(3).float(), requires_grad=False)

        selfconh = torch.Tensor(
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, -1]
            ]
        )
        self.selfconh = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.selfconh.weight = torch.nn.Parameter(selfconh.unsqueeze(0).unsqueeze(0).float(), requires_grad=False)
        selfconv = torch.Tensor(
            [[0, 0, 0],
             [0, 0, 1],
             [0, -1, 0]
            ]
        )
        self.selfconv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.selfconv.weight = torch.nn.Parameter(selfconv.unsqueeze(0).unsqueeze(0).float(), requires_grad=False)

        selfconhIndW = torch.Tensor(
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 1]
            ]
        )
        self.selfconhInd = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.selfconhInd.weight = torch.nn.Parameter(selfconhIndW.unsqueeze(0).unsqueeze(0).float(), requires_grad=False)
        selfconvIndW = torch.Tensor(
            [[0, 0, 0],
             [0, 0, 1],
             [0, 1, 0]
            ]
        )
        self.selfconvInd = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.selfconvInd.weight = torch.nn.Parameter(selfconvIndW.unsqueeze(0).unsqueeze(0).float(), requires_grad=False)

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

        weightsxval = torch.Tensor([[0., 0., 0.],
                                    [0., 1., 1.],
                                    [0., 0., 0.]]).unsqueeze(0).unsqueeze(0)

        weightsyval = torch.Tensor([[0., 0., 0.],
                                    [0., 1., 0.],
                                    [0., 1., 0.]]).unsqueeze(0).unsqueeze(0)
        self.diffx_sharp_val = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.diffy_sharp_val = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.diffx_sharp_val.weight = nn.Parameter(weightsxval, requires_grad=False)
        self.diffy_sharp_val.weight = nn.Parameter(weightsyval, requires_grad=False)

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

        # vind = 1
        # tensor2disp(surfnorm[:, 0:1, :, :] + 1, vmax=2, ind=vind).show()
        # tensor2disp(surfnorm[:, 1:2, :, :] + 1, vmax=2, ind=vind).show()
        # tensor2disp(surfnorm[:, 2:3, :, :] + 1, vmax=2, ind=vind).show()
        # tensor2rgb((surfnorm + 1) / 2, ind=vind).show()

        return surfnorm

    def depth2ang(self, depthMap, intrinsic, issharp=True):
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

        a = (self.yy - by) / fy * vx2 + vx3
        b = -vx1

        u = vy3 + (self.xx - bx) / fx * vy1
        v = -vy2

        angh = torch.atan2(a, -b)
        angv = torch.atan2(u, -v)

        angh = angh.unsqueeze(1)
        angv = angv.unsqueeze(1)

        ang = torch.cat([angh, angv], dim=1)

        # tensor2disp(angh + np.pi, vmax=2*np.pi, ind=0).show()
        # tensor2disp(angv + np.pi, vmax=2*np.pi, ind=0).show()
        return ang

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

    def ang2edge(self, ang, intrinsic):
        protectmin = 1e-7

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

        low_angh = torch.atan2(-a1, b1)
        high_angh = torch.atan2(a2, -b2)
        low_angv = torch.atan2(-u1, v1)
        high_angv = torch.atan2(u2, -v2)

        outboundh = ((angh > high_angh) * (angh < low_angh)).unsqueeze(1)
        outboundv = ((angv > high_angv) * (angv < low_angv)).unsqueeze(1)

        logh = torch.log(torch.clamp(torch.abs(a3 * b1 - a1 * b3), min=protectmin)) - torch.log(torch.clamp(torch.abs(a3 * b2 - a2 * b3), min=protectmin))
        logv = torch.log(torch.clamp(torch.abs(u3 * v1 - u1 * v3), min=protectmin)) - torch.log(torch.clamp(torch.abs(u3 * v2 - u2 * v3), min=protectmin))
        logh = logh.unsqueeze(1)
        logv = logv.unsqueeze(1)

        edge = (torch.abs(logh) > 0.05) + (torch.abs(logv) > 0.05) + outboundh + outboundv
        edge = edge.int()
        # tensor2disp(torch.abs(logh) > 0.1, vmax=0.1, ind=0).show()
        # tensor2disp(torch.abs(logv) > 0.1, vmax=0.1, ind=0).show()
        # tensor2disp(edge, vmax=0.1, ind=0).show()
        return edge

    def depth2ang_log(self, depthMap, intrinsic):
        depthMaps = depthMap.squeeze(1)
        fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        bx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        by = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])

        p2dhx = (self.xx - bx) / fx * depthMaps
        p2dhy = (((self.yy - by) / fy) ** 2 + 1) * depthMaps

        p2dvx = (self.yy - by) / fy * depthMaps
        p2dvy = (((self.xx - bx) / fx) ** 2 + 1) * depthMaps

        angh = torch.atan2(self.diffx_sharp(p2dhy.unsqueeze(1)), self.diffx_sharp(p2dhx.unsqueeze(1)))
        angv = torch.atan2(self.diffy_sharp(p2dvy.unsqueeze(1)), self.diffy_sharp(p2dvx.unsqueeze(1)))

        ang = torch.cat([angh, angv], dim=1)

        # ang2 = self.depth2ang(depthMap, intrinsic, True)
        # tensor2disp(angh + np.pi, vmax=np.pi * 2, ind=0).show()
        # tensor2disp(ang2[:,0:1,:,:] + np.pi, vmax=np.pi * 2, ind=0).show()
        # tensor2disp(angv + np.pi, vmax=np.pi * 2, ind=0).show()
        # tensor2disp(ang2[:,1:2,:,:] + np.pi, vmax=np.pi * 2, ind=0).show()
        #
        # log = self.ang2log(intrinsic, ang)
        # logh = log[:, 0, :, :]
        # logv = log[:, 1, :, :]
        #
        # import random
        # ckx = random.randint(0, self.width)
        # cky = random.randint(0, self.height)
        # ckz = random.randint(0, self.batch_size - 1)
        #
        # ckhgtl = torch.log(depthMap[ckz, 0, cky, ckx + 1]) - torch.log(depthMap[ckz, 0, cky, ckx])
        # ckhestl = logh[ckz, cky, ckx]
        #
        # ckvgtl = torch.log(depthMap[ckz, 0, cky + 1, ckx]) - torch.log(depthMap[ckz, 0, cky, ckx])
        # ckvestl = logv[ckz, cky, ckx]
        return ang

    def intergrationloss_ang(self, ang, intrinsic, depthMap):
        anglebound = 0.1
        protectmin = 1e-6
        vlossw = self.vlossw
        angw = self.angw
        sclw = self.sclw

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

        depthMapl = torch.log(torch.clamp(depthMap, min=protectmin))

        low_angh = torch.atan2(-a1, b1)
        high_angh = torch.atan2(a2, -b2)
        pred_angh = angh
        inboundh = ((pred_angh < (high_angh - anglebound)) * (pred_angh > (low_angh + anglebound))).float()

        low_angv = torch.atan2(-u1, v1)
        high_angv = torch.atan2(u2, -v2)
        pred_angv = angv
        inboundv = ((pred_angv < (high_angv - anglebound)) * (pred_angv > (low_angv + anglebound))).float()

        logh = torch.log(torch.clamp(torch.abs(a3 * b1 - a1 * b3), min=protectmin)) - torch.log(torch.clamp(torch.abs(a3 * b2 - a2 * b3), min=protectmin))
        logh = torch.clamp(logh, min=-10, max=10)

        logv = torch.log(torch.clamp(torch.abs(u3 * v1 - u1 * v3), min=protectmin)) - torch.log(torch.clamp(torch.abs(u3 * v2 - u2 * v3), min=protectmin))
        logv = torch.clamp(logv, min=-10, max=10)

        logh = logh.unsqueeze(1)
        logv = logv.unsqueeze(1)
        inboundh = inboundh.unsqueeze(1)
        inboundv = inboundv.unsqueeze(1)

        vallidarmask = (depthMap > 0).float()

        inth = self.inth(logh)
        gth = self.gth(depthMapl)
        indh = ((self.idh(vallidarmask) == 2) * (self.inth(1-inboundh) == 0)).float()
        hloss1 = torch.sum(torch.abs(gth - inth) * indh) / (torch.sum(indh) + 1)

        intv = self.intv(logv)
        gtv = self.gtv(depthMapl)
        indv = ((self.idv(vallidarmask) == 2) * (self.intv(1-inboundv) == 0)).float()
        vloss1 = torch.sum(torch.abs(gtv - intv) * indv) * vlossw / (torch.sum(indv) + 1)

        scl_pixelwise = self.selfconh(logh) + self.selfconv(logv)
        scl = torch.mean(torch.abs(scl_pixelwise))

        loss = hloss1 + vloss1 + scl * sclw
        return loss, hloss1, vloss1, torch.sum((1 - inboundh)), torch.sum((1 - inboundv))

    def intergrationloss_ang_validation(self, ang, intrinsic, depthMap):
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

        depthMapl = torch.log(torch.clamp(depthMap, min=protectmin))

        logh = torch.log(torch.clamp(torch.abs(a3 * b1 - a1 * b3), min=protectmin)) - torch.log(torch.clamp(torch.abs(a3 * b2 - a2 * b3), min=protectmin))
        logh = torch.clamp(logh, min=-10, max=10)

        logv = torch.log(torch.clamp(torch.abs(u3 * v1 - u1 * v3), min=protectmin)) - torch.log(torch.clamp(torch.abs(u3 * v2 - u2 * v3), min=protectmin))
        logv = torch.clamp(logv, min=-10, max=10)

        logh = logh.unsqueeze(1)
        logv = logv.unsqueeze(1)

        vallidarmask = (depthMap > 0).float()

        inth = self.inth(logh)
        gth = self.gth(depthMapl)
        indh = (self.idh(vallidarmask) == 2).float()
        hloss = torch.sum(torch.abs(gth - inth) * indh) / (torch.sum(indh) + 1)

        intv = self.intv(logv)
        gtv = self.gtv(depthMapl)
        indv = (self.idv(vallidarmask) == 2).float()
        vloss = torch.sum(torch.abs(gtv - intv) * indv) / (torch.sum(indv) + 1)

        loss = hloss + vloss

        return loss

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

        # tensor2rgb((surfacenormal + 1) / 2, ind=0).show()
        # tensor2rgb((self.depth2norm(depthMap, intrinsic) + 1) / 2, ind=0).show()

        return surfacenormal

    def ang2dirs(self, ang, intrinsic):
        fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        bx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        by = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])

        angh = ang[:, 0, :, :]
        angv = ang[:, 1, :, :]

        normx = torch.stack([torch.cos(angh), (self.yy - by) / fy * torch.sin(angh), torch.sin(angh)], dim=1)
        normy = torch.stack([(self.xx - bx) / fx * torch.sin(angv), torch.cos(angv), torch.sin(angv)], dim=1)

        normx = F.normalize(normx, dim=1)
        normy = F.normalize(normy, dim=1)

        return normx, normy

    def normal2ang(self, surfnorm, intrinsic):
        fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        bx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        by = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])

        surfnormx = torch.stack([surfnorm[:, 1, :, :] * (self.yy - by) / fy + surfnorm[:, 2, :, :], -(self.yy - by) / fy * surfnorm[:, 0, :, :], -surfnorm[:, 0, :, :]], dim=1)
        surfnormy = torch.stack([-surfnorm[:, 1, :, :] * (self.xx - bx) / fx, (self.xx - bx) / fx * surfnorm[:, 0, :, :] + surfnorm[:, 2, :, :], -surfnorm[:, 1, :, :]], dim=1)

        a3 = (self.yy - by) / fy * surfnormx[:, 1, :, :] + surfnormx[:, 2, :, :]
        b3 = -surfnormx[:, 0, :, :]

        u3 = surfnormy[:, 2, :, :] + (self.xx - bx) / fx * surfnormy[:, 0, :, :]
        v3 = -surfnormy[:, 1, :, :]

        pred_angh = torch.atan2(a3, -b3).unsqueeze(1)
        pred_angv = torch.atan2(u3, -v3).unsqueeze(1)

        predang = torch.cat([pred_angh, pred_angv], dim=1)

        return predang

    def ang2grad(self, ang, intrinsic, depthMap):
        with torch.no_grad():
            anglebound = 0.1

            fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
            bx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
            fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
            by = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])

            angh = ang[:, 0, :, :]
            angv = ang[:, 1, :, :]

            a1 = ((self.yy - by) / fy)**2 + 1
            b1 = -(self.xx - bx) / fx

            a2 = ((self.yy - by) / fy)**2 + 1
            b2 = -(self.xx + 1 - bx) / fx

            u1 = ((self.xx - bx) / fx)**2 + 1
            v1 = -(self.yy - by) / fy

            u2 = ((self.xx - bx) / fx)**2 + 1
            v2 = -(self.yy + 1 - by) / fy

            low_angh = torch.atan2(-a1, b1)
            high_angh = torch.atan2(a2, -b2)
            pred_angh = angh
            inboundh = ((pred_angh < (high_angh - anglebound)) * (pred_angh > (low_angh + anglebound))).float()
            inboundh = inboundh.unsqueeze(1)

            low_angv = torch.atan2(-u1, v1)
            high_angv = torch.atan2(u2, -v2)
            pred_angv = angv
            inboundv = ((pred_angv < (high_angv - anglebound)) * (pred_angv > (low_angv + anglebound))).float()
            inboundv = inboundv.unsqueeze(1)

            kx = (((self.yy - by) / fy) ** 2 + 1) * torch.cos(angh) - (self.xx - bx) / fx * torch.sin(angh)
            signx = torch.sign(kx)
            kx = signx * torch.clamp(torch.abs(kx), min=1e-3, max=1e3)
            kx = torch.sin(angh) / kx / fx

            ky = (((self.xx - bx) / fx) ** 2 + 1) * torch.cos(angv) - (self.yy - by) / fy * torch.sin(angv)
            signy = torch.sign(ky)
            ky = signy * torch.clamp(torch.abs(ky), min=1e-3, max=1e3)
            ky = torch.sin(angv) / ky / fy

            if torch.sum(torch.isnan(kx)) + torch.sum(torch.isnan(ky)) > 0:
                print("error detected")
                return -1

        depthMaps = depthMap.squeeze(1)
        depthMap_gradx_est = (depthMaps * kx).unsqueeze(1)
        depthMap_grady_est = (depthMaps * ky).unsqueeze(1)

        # depthMap_gradx_est = depthMaps / fx * torch.sin(angh) / ((((self.yy - by) / fy) ** 2 + 1) * torch.cos(angh) - (self.xx - bx) / fx * torch.sin(angh))
        # depthMap_grady_est = depthMaps / fy * torch.sin(angv) / ((((self.xx - bx) / fx) ** 2 + 1) * torch.cos(angv) - (self.yy - by) / fy * torch.sin(angv))
        #
        # depthMap_gradx_est = depthMap_gradx_est.unsqueeze(1).clamp(min=-1e6, max=1e6)
        # depthMap_grady_est = depthMap_grady_est.unsqueeze(1).clamp(min=-1e6, max=1e6)

        # Check
        # depthMap_gradx = self.diffx_sharp(depthMap)
        # depthMap_grady = self.diffy_sharp(depthMap)
        #
        # tensor2grad(depthMap_gradx_est, viewind=0, percentile=80).show()
        # tensor2grad(depthMap_gradx, viewind=0, percentile=80).show()
        #
        # tensor2grad(depthMap_grady_est, viewind=0, percentile=80).show()
        # tensor2grad(depthMap_grady, viewind=0, percentile=80).show()
        return depthMap_gradx_est, depthMap_grady_est, inboundh, inboundv

    def depth2grad(self, depthMap, isgt=False):
        depthMap_gradx = self.diffx_sharp(depthMap)
        depthMap_grady = self.diffy_sharp(depthMap)

        if isgt:
            depthMapInd = (depthMap > 0).float()
            depthMap_gradx_ind = self.diffx_sharp_val(depthMapInd)
            depthMap_grady_ind = self.diffy_sharp_val(depthMapInd)

            depthMap_gradx[depthMap_gradx_ind != 2] = 0
            depthMap_grady[depthMap_grady_ind != 2] = 0

        return depthMap_gradx, depthMap_grady