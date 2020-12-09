import torch
import shapeintegration_cuda
import torch.nn as nn

class IntegrationCRFFunction(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __int__(self):
        super(IntegrationCRFFunction, self).__init__()

    @staticmethod
    def forward(ctx, pred_log, mask, variance, depthin, clipvariance, maxrange):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        pred_log = pred_log.contiguous().float()
        mask = mask.contiguous().int()
        variance = variance.contiguous().float()
        depthin = depthin.contiguous().float()

        bz, _, h, w = depthin.shape
        clipvariance = float(clipvariance)
        maxrange = float(maxrange)

        depthout = torch.zeros_like(depthin)
        summedconfidence = torch.zeros_like(depthin)
        sdirintexpvariance = torch.zeros([bz, 4, h, w], dtype=torch.float32, device=depthin.device)
        sdirintweighteddepth = torch.zeros([bz, 4, h, w], dtype=torch.float32, device=depthin.device)
        sdirlateralre = torch.zeros([bz, 4, h, w], dtype=torch.float32, device=depthin.device)

        shapeintegration_cuda.shapeIntegration_crf_variance_forward(pred_log, mask, variance, depthin, depthout, summedconfidence, sdirintexpvariance, sdirintweighteddepth, sdirlateralre, h, w, bz, clipvariance, maxrange)

        ctx.save_for_backward(pred_log, mask, variance, depthin, depthout, summedconfidence, sdirintexpvariance, sdirintweighteddepth, sdirlateralre)
        ctx.h = h
        ctx.w = w
        ctx.bz = bz
        ctx.clipvariance = clipvariance
        ctx.maxrange = maxrange
        return depthout

    @staticmethod
    def backward(ctx, grad_depthin):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_depthin = grad_depthin.contiguous().float()

        pred_log, mask, variance, depthin, depthout, summedconfidence, sdirintexpvariance, sdirintweighteddepth, sdirlateralre = ctx.saved_tensors
        h = ctx.h
        w = ctx.w
        bz = ctx.bz
        clipvariance = ctx.clipvariance
        maxrange = ctx.maxrange

        grad_varianceout = torch.zeros_like(variance)
        grad_depthout = torch.zeros_like(depthin)
        grad_logout = torch.zeros_like(pred_log)

        shapeintegration_cuda.shapeIntegration_crf_variance_backward(pred_log, mask, variance, depthin, depthout, summedconfidence, sdirintexpvariance, sdirintweighteddepth, sdirlateralre, grad_depthin, grad_varianceout, grad_depthout, grad_logout, h, w, bz, clipvariance, maxrange)
        return grad_logout, None, grad_varianceout, grad_depthout, None, None

class CRFIntegrationModule(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, clipvariance, maxrange):
        super(CRFIntegrationModule, self).__init__()

        self.intfunc = IntegrationCRFFunction.apply
        self.clipvariance = clipvariance
        self.maxrange = maxrange

    def forward(self, pred_log, mask, variance, depthin, lam, times=1):
        depthout = depthin.clone()
        for k in range(times):
            lateralre = self.intfunc(pred_log, mask, variance, depthout, self.clipvariance, self.maxrange)
            optselector = (lateralre > 0).float()
            depthout = (1 - optselector) * depthin + optselector * (depthin * (1 - lam) + lateralre * lam)
        return depthout

    def compute_lateralre(self, pred_log, mask, variance, depthin):
        lateralre = self.intfunc(pred_log, mask, variance, depthin, self.clipvariance, self.maxrange)
        optselector = (lateralre > 0).float()
        depthout = (1 - optselector) * depthin + optselector * lateralre
        return depthout