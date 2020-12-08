#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <math_constants.h>

namespace {

}
__global__ void shapeIntegration_crf_variance_forward_cuda_kernel(
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> log,
    const torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> mask,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> variance,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> depth_optedin,
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> depth_optedout,
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> summedconfidence,
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> sdirintexpvariance,
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> sdirintweighteddepth,
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> sdirlateralre,
    const int height,
    const int width,
    const int bs,
    const float clipvariance,
    const float maxrange
    ) {
    int m;
    int n;

    int sm;
    int sn;

    float intlog;
    float intvariance;
    float intexpvariance;

    for(int i = threadIdx.x; i < height * width; i = i + blockDim.x){
        m = i / width;
        n = i - m * width;

        if(mask[blockIdx.x][0][m][n] == 1){
            intexpvariance = 0;

            // Left
            sm = m;
            intvariance = 0;
            for(sn = n-1; sn >= 0; sn--){
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(n - sn > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intvariance += variance[blockIdx.x][0][sm][sn];
                intexpvariance += exp(-intvariance);
            }

            // Right
            sm = m;
            intvariance = 0;
            for(sn = n+1; sn < width; sn++){
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(sn - n > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intvariance += variance[blockIdx.x][0][sm][sn];
                intexpvariance += exp(-intvariance);
            }

            // Up
            sn = n;
            intvariance = 0;
            for(sm = m-1; sm >= 0; sm--){
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(m - sm > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intvariance += variance[blockIdx.x][0][sm][sn];
                intexpvariance += exp(-intvariance);
            }

            // Down
            sn = n;
            intvariance = 0;
            for(sm = m+1; sm < height; sm++){
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(sm - m > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intvariance += variance[blockIdx.x][0][sm][sn];
                intexpvariance += exp(-intvariance);
            }

            summedconfidence[blockIdx.x][0][m][n] = intexpvariance;

            // Left
            sm = m;
            intlog = 0;
            intvariance = 0;
            for(sn = n-1; sn >= 0; sn--){
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(n - sn > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intlog += -log[blockIdx.x][0][sm][sn];
                intvariance += variance[blockIdx.x][0][sm][sn];
                sdirintexpvariance[blockIdx.x][0][m][n] += exp(-intvariance);
                sdirintweighteddepth[blockIdx.x][0][m][n] += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance);
                sdirlateralre[blockIdx.x][0][m][n] += -exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance) / intexpvariance;
            }

            // Right
            sm = m;
            intlog = 0;
            intvariance = 0;
            for(sn = n+1; sn < width; sn++){
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(sn - n > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intlog += log[blockIdx.x][0][sm][sn-1];
                intvariance += variance[blockIdx.x][0][sm][sn];
                sdirintexpvariance[blockIdx.x][1][m][n] += exp(-intvariance);
                sdirintweighteddepth[blockIdx.x][1][m][n] += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance);
                sdirlateralre[blockIdx.x][1][m][n] += -exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance) / intexpvariance;
            }

            // Up
            sn = n;
            intlog = 0;
            intvariance = 0;
            for(sm = m-1; sm >= 0; sm--){
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(m - sm > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intlog += -log[blockIdx.x][1][sm][sn];
                intvariance += variance[blockIdx.x][0][sm][sn];
                sdirintexpvariance[blockIdx.x][2][m][n] += exp(-intvariance);
                sdirintweighteddepth[blockIdx.x][2][m][n] += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance);
                sdirlateralre[blockIdx.x][2][m][n] += -exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance) / intexpvariance;
            }

            // Down
            sn = n;
            intlog = 0;
            intvariance = 0;
            for(sm = m+1; sm < height; sm++){
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(sm - m > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intlog += log[blockIdx.x][1][sm-1][sn];
                intvariance += variance[blockIdx.x][0][sm][sn];
                sdirintexpvariance[blockIdx.x][3][m][n] += exp(-intvariance);
                sdirintweighteddepth[blockIdx.x][3][m][n] += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance);
                sdirlateralre[blockIdx.x][3][m][n] += -exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance) / intexpvariance;
            }

            depth_optedout[blockIdx.x][0][m][n] = -(sdirlateralre[blockIdx.x][0][m][n] + sdirlateralre[blockIdx.x][1][m][n] + sdirlateralre[blockIdx.x][2][m][n] + sdirlateralre[blockIdx.x][3][m][n]);

        }

    }
    return;
    }

__global__ void shapeIntegration_crf_variance_backward_cuda_kernel(
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> log,
    const torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> mask,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> variance,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> depth_optedin,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> depth_optedout,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> summedconfidence,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> sdirintexpvariance,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> sdirintweighteddepth,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> sdirlateralre,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> grad_depthin,
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> grad_varianceout,
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> grad_depthout,
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> grad_logout,
    const int height,
    const int width,
    const int bs,
    const float clipvariance,
    const float maxrange
    ) {

    int m;
    int n;

    int sm;
    int sn;

    float intlog;
    float intvariance;
    float intexpvariance;
    float lateralre;

    float cursdirintexpvariance;
    float cursdirweighteddepth;
    float cursdirlateralre;

    for(int i = threadIdx.x; i < height * width; i = i + blockDim.x){
        m = i / width;
        n = i - m * width;

        if(mask[blockIdx.x][0][m][n] == 1){
            intexpvariance = summedconfidence[blockIdx.x][0][m][n];
            lateralre = depth_optedout[blockIdx.x][0][m][n];

            // Left
            sm = m;
            intlog = 0;
            intvariance = 0;
            cursdirintexpvariance = 0;
            cursdirweighteddepth = 0;
            cursdirlateralre = 0;
            for(sn = n-1; sn >= 0; sn--){
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(n - sn > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intlog += -log[blockIdx.x][0][sm][sn];
                intvariance += variance[blockIdx.x][0][sm][sn];

                atomicAdd((float*)&grad_varianceout[blockIdx.x][0][sm][sn], grad_depthin[blockIdx.x][0][m][n] * ((sdirintexpvariance[blockIdx.x][0][m][n] - cursdirintexpvariance) / intexpvariance * lateralre - (sdirintweighteddepth[blockIdx.x][0][m][n] - cursdirweighteddepth) / intexpvariance));
                atomicAdd((float*)&grad_logout[blockIdx.x][0][sm][sn], -(sdirlateralre[blockIdx.x][0][m][n] - cursdirlateralre) * grad_depthin[blockIdx.x][0][m][n]);
                cursdirintexpvariance += exp(-intvariance);
                cursdirweighteddepth += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance);
                cursdirlateralre += -exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance) / intexpvariance;

                atomicAdd((float*)&grad_depthout[blockIdx.x][0][sm][sn], grad_depthin[blockIdx.x][0][m][n] * exp(-intlog) * exp(-intvariance) / intexpvariance);
            }

            // Right
            sm = m;
            intlog = 0;
            intvariance = 0;
            cursdirintexpvariance = 0;
            cursdirweighteddepth = 0;
            cursdirlateralre = 0;
            for(sn = n+1; sn < width; sn++){
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(sn - n > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intlog += log[blockIdx.x][0][sm][sn-1];
                intvariance += variance[blockIdx.x][0][sm][sn];

                atomicAdd((float*)&grad_varianceout[blockIdx.x][0][sm][sn], grad_depthin[blockIdx.x][0][m][n] * ((sdirintexpvariance[blockIdx.x][1][m][n] - cursdirintexpvariance) / intexpvariance * lateralre - (sdirintweighteddepth[blockIdx.x][1][m][n] - cursdirweighteddepth) / intexpvariance));
                atomicAdd((float*)&grad_logout[blockIdx.x][0][sm][sn-1], (sdirlateralre[blockIdx.x][1][m][n] - cursdirlateralre) * grad_depthin[blockIdx.x][0][m][n]);
                cursdirintexpvariance += exp(-intvariance);
                cursdirweighteddepth += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance);
                cursdirlateralre += -exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance) / intexpvariance;

                atomicAdd((float*)&grad_depthout[blockIdx.x][0][sm][sn], grad_depthin[blockIdx.x][0][m][n] * exp(-intlog) * exp(-intvariance) / intexpvariance);
            }

            // Up
            sn = n;
            intlog = 0;
            intvariance = 0;
            cursdirintexpvariance = 0;
            cursdirweighteddepth = 0;
            cursdirlateralre = 0;
            for(sm = m-1; sm >= 0; sm--){
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(m - sm > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intlog += -log[blockIdx.x][1][sm][sn];
                intvariance += variance[blockIdx.x][0][sm][sn];

                atomicAdd((float*)&grad_varianceout[blockIdx.x][0][sm][sn], grad_depthin[blockIdx.x][0][m][n] * ((sdirintexpvariance[blockIdx.x][2][m][n] - cursdirintexpvariance) / intexpvariance * lateralre - (sdirintweighteddepth[blockIdx.x][2][m][n] - cursdirweighteddepth) / intexpvariance));
                atomicAdd((float*)&grad_logout[blockIdx.x][1][sm][sn], -(sdirlateralre[blockIdx.x][2][m][n] - cursdirlateralre) * grad_depthin[blockIdx.x][0][m][n]);
                cursdirintexpvariance += exp(-intvariance);
                cursdirweighteddepth += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance);
                cursdirlateralre += -exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance) / intexpvariance;

                atomicAdd((float*)&grad_depthout[blockIdx.x][0][sm][sn], grad_depthin[blockIdx.x][0][m][n] * exp(-intlog) * exp(-intvariance) / intexpvariance);
            }

            // Down
            sn = n;
            intlog = 0;
            intvariance = 0;
            cursdirintexpvariance = 0;
            cursdirweighteddepth = 0;
            cursdirlateralre = 0;
            for(sm = m+1; sm < height; sm++){
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(sm - m > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intlog += log[blockIdx.x][1][sm-1][sn];
                intvariance += variance[blockIdx.x][0][sm][sn];

                atomicAdd((float*)&grad_varianceout[blockIdx.x][0][sm][sn], grad_depthin[blockIdx.x][0][m][n] * ((sdirintexpvariance[blockIdx.x][3][m][n] - cursdirintexpvariance) / intexpvariance * lateralre - (sdirintweighteddepth[blockIdx.x][3][m][n] - cursdirweighteddepth) / intexpvariance));
                atomicAdd((float*)&grad_logout[blockIdx.x][1][sm-1][sn], (sdirlateralre[blockIdx.x][3][m][n] - cursdirlateralre) * grad_depthin[blockIdx.x][0][m][n]);
                cursdirintexpvariance += exp(-intvariance);
                cursdirweighteddepth += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance);
                cursdirlateralre += -exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance) / intexpvariance;

                atomicAdd((float*)&grad_depthout[blockIdx.x][0][sm][sn], grad_depthin[blockIdx.x][0][m][n] * exp(-intlog) * exp(-intvariance) / intexpvariance);
            }
        }

    }
    return;

    }

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
    ){
      const int threads = 512;
      shapeIntegration_crf_variance_forward_cuda_kernel<<<bs, threads>>>(
            log.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            mask.packed_accessor<int,4,torch::RestrictPtrTraits,size_t>(),
            variance.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            depth_optedin.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            depth_optedout.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            summedconfidence.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            sdirintexpvariance.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            sdirintweighteddepth.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            sdirlateralre.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            height,
            width,
            bs,
            clipvariance,
            maxrange
            );

    return;
    }

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
    ){
      const int threads = 512;
      shapeIntegration_crf_variance_backward_cuda_kernel<<<bs, threads>>>(
            log.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            mask.packed_accessor<int,4,torch::RestrictPtrTraits,size_t>(),
            variance.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            depth_optedin.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            depth_optedout.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            summedconfidence.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            sdirintexpvariance.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            sdirintweighteddepth.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            sdirlateralre.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            grad_depthin.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            grad_varianceout.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            grad_depthout.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            grad_logout.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            height,
            width,
            bs,
            clipvariance,
            maxrange
            );

    return;
    }


