#include "baseline/flash_attn_v2_baseline.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

static cublasHandle_t get_handle()
{
    static cublasHandle_t h = []{
        cublasHandle_t tmp; cublasCreate(&tmp); return tmp;
    }();
    return h;
}

// -device 版 row-wise softmax，极简 inline
static __global__ void softmax_inplace_dev(float* s, int BH, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BH * N) return;
    int row = idx / N;
    int i   = idx % N;
    float* rowPtr = s + row * N;

    // max
    float max_val = -1e38f;
    for (int j = 0; j < N; ++j) max_val = fmaxf(max_val, rowPtr[j]);
    // exp & sum
    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        float ev = expf(rowPtr[j] - max_val);
        rowPtr[j] = ev;
        sum += ev;
    }
    // normalize
    for (int j = 0; j < N; ++j) rowPtr[j] /= sum;
}

void flash_attn_v2_cublas_fwd(const float* q, const float* k, const float* v,
                              float* out,
                              int B, int H, int N, int D)
{
    cublasHandle_t handle = get_handle();
    const float alpha = 1.0f, beta = 0.0f;
    size_t strideQD = N * D;

    // 1) S = Q K^T    (B*H 个 N×N)
    float* s = nullptr;
    cudaMalloc(&s, B * H * N * N * sizeof(float));
    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, N, D,
        &alpha,
        k, CUDA_R_32F, D, strideQD,
        q, CUDA_R_32F, D, strideQD,
        &beta,
        s, CUDA_R_32F, N, N * N,
        B * H,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // 2) row-wise softmax
    dim3 block(128);
    dim3 grid((B * H * N + block.x - 1) / block.x);
    softmax_inplace_dev<<<grid, block>>>(s, B * H, N);
    cudaDeviceSynchronize();

    // 3) O = S V
    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        D, N, N,
        &alpha,
        v, CUDA_R_32F, D, strideQD,
        s, CUDA_R_32F, N, N * N,
        &beta,
        out, CUDA_R_32F, D, strideQD,
        B * H,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cudaFree(s);
}