#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

// CUDA forward declarations
void shapeIntegration_crf_variance_forward_cuda(
    torch::Tensor log,
    torch::Tensor mask,
    torch::Tensor variance,
    torch::Tensor depth_optedin,
    torch::Tensor depth_optedout,
    torch::Tensor summedconfidence,
    torch::Tensor sdirintexpvariance,
    torch::Tensor sdirintweighteddepth,
    torch::Tensor sdirlateralre,
    int height,
    int width,
    int bs,
    float clipvariance,
    float maxrange
    );

void shapeIntegration_crf_variance_backward_cuda(
    torch::Tensor log,
    torch::Tensor mask,
    torch::Tensor variance,
    torch::Tensor depth_optedin,
    torch::Tensor depth_optedout,
    torch::Tensor summedconfidence,
    torch::Tensor sdirintexpvariance,
    torch::Tensor sdirintweighteddepth,
    torch::Tensor sdirlateralre,
    torch::Tensor grad_depthin,
    torch::Tensor grad_varianceout,
    torch::Tensor grad_depthout,
    torch::Tensor grad_logout,
    int height,
    int width,
    int bs,
    float clipvariance,
    float maxrange
    );

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//torch::Tensor bnmorph_find_coorespond_pts(
void shapeIntegration_crf_variance_forward(
    torch::Tensor log,
    torch::Tensor mask,
    torch::Tensor variance,
    torch::Tensor depth_optedin,
    torch::Tensor depth_optedout,
    torch::Tensor summedconfidence,
    torch::Tensor sdirintexpvariance,
    torch::Tensor sdirintweighteddepth,
    torch::Tensor sdirlateralre,
    int height,
    int width,
    int bs,
    float clipvariance,
    float maxrange
    ) {
    CHECK_INPUT(log);
    CHECK_INPUT(mask);
    CHECK_INPUT(variance);
    CHECK_INPUT(depth_optedin);
    CHECK_INPUT(depth_optedout);
    CHECK_INPUT(summedconfidence);
    CHECK_INPUT(sdirintexpvariance);
    CHECK_INPUT(sdirintweighteddepth);
    CHECK_INPUT(sdirlateralre);
    shapeIntegration_crf_variance_forward_cuda(log, mask, variance, depth_optedin, depth_optedout, summedconfidence, sdirintexpvariance, sdirintweighteddepth, sdirlateralre, height, width, bs, clipvariance, maxrange);
    return;
}

void shapeIntegration_crf_variance_backward(
    torch::Tensor log,
    torch::Tensor mask,
    torch::Tensor variance,
    torch::Tensor depth_optedin,
    torch::Tensor depth_optedout,
    torch::Tensor summedconfidence,
    torch::Tensor sdirintexpvariance,
    torch::Tensor sdirintweighteddepth,
    torch::Tensor sdirlateralre,
    torch::Tensor grad_depthin,
    torch::Tensor grad_varianceout,
    torch::Tensor grad_depthout,
    torch::Tensor grad_logout,
    int height,
    int width,
    int bs,
    float clipvariance,
    float maxrange
    ) {
    CHECK_INPUT(log);
    CHECK_INPUT(mask);
    CHECK_INPUT(variance);
    CHECK_INPUT(depth_optedin);
    CHECK_INPUT(depth_optedout);
    CHECK_INPUT(summedconfidence);
    CHECK_INPUT(sdirintexpvariance);
    CHECK_INPUT(sdirintweighteddepth);
    CHECK_INPUT(sdirlateralre);
    CHECK_INPUT(grad_depthin);
    CHECK_INPUT(grad_varianceout);
    CHECK_INPUT(grad_depthout);
    CHECK_INPUT(grad_logout);
    shapeIntegration_crf_variance_backward_cuda(log, mask, variance, depth_optedin, depth_optedout, summedconfidence, sdirintexpvariance, sdirintweighteddepth, sdirlateralre, grad_depthin, grad_varianceout, grad_depthout, grad_logout, height, width, bs, clipvariance, maxrange);
    return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("shapeIntegration_crf_variance_forward", &shapeIntegration_crf_variance_forward, "crf based shape integration with confidence forward function");
  m.def("shapeIntegration_crf_variance_backward", &shapeIntegration_crf_variance_backward, "crf based shape integration with confidence backward function");
}
