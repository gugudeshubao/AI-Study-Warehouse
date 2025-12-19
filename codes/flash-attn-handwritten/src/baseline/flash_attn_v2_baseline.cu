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


static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t h = []{
        cublasHandle_t tmp; cublasCreate(&tmp); return tmp;
    }();
    return h;
}

// 设备端行方向 softmax 反向
__global__ void softmax_backward_row_device(float* grad, const float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    extern __shared__ float shared[];
    float* s_sum = shared;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sum += grad[i] * out[i];
    s_sum[threadIdx.x] = sum;
    __syncthreads();
    // 归约求和
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) s_sum[0] = sum;
    __syncthreads();
    sum = s_sum[0];
    grad[idx] = out[idx] * (grad[idx] - sum);
}

FlashAttnGrad cublas_flash_v2_backward(const float* q, const float* k, const float* v,
                                       const float* out, const float* dout,
                                       int B, int H, int N, int D)
{
    size_t len = B * H * N * D;
    float *dQ, *dK, *dV;
    cudaMalloc(&dQ, len * sizeof(float));
    cudaMalloc(&dK, len * sizeof(float));
    cudaMalloc(&dV, len * sizeof(float));
    cudaMemset(dQ, 0, len * sizeof(float));
    cudaMemset(dK, 0, len * sizeof(float));
    cudaMemset(dV, 0, len * sizeof(float));

    cublasHandle_t handle = get_cublas_handle();
    const float alpha = 1.0f, beta = 0.0f;

    // 临时设备缓冲区
    float *s, *p, *grad_s;
    cudaMalloc(&s, B * H * N * N * sizeof(float));
    cudaMalloc(&p, B * H * N * N * sizeof(float));
    cudaMalloc(&grad_s, B * H * N * N * sizeof(float));

    // ===== 1) 前向 S = Q@K^T =====
    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, N, D,
        &alpha,
        k, CUDA_R_32F, D, N * D,
        q, CUDA_R_32F, D, N * D,
        &beta,
        s, CUDA_R_32F, N, N * N,
        B * H,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // ===== 2) P = softmax(S) =====
    dim3 block(128);
    for (int bh = 0; bh < B * H; ++bh) {
        for (int i = 0; i < N; ++i) {
            // 行 max
            float maxVal = -1e38f;
            for (int j = 0; j < N; ++j)
                maxVal = fmaxf(maxVal, s[bh * N * N + i * N + j]);
            // 行 exp & sum
            float sum = 0.0f;
            for (int j = 0; j < N; ++j) {
                float ev = expf(s[bh * N * N + i * N + j] - maxVal);
                p[bh * N * N + i * N + j] = ev;
                sum += ev;
            }
            // 行 normalize
            for (int j = 0; j < N; ++j)
                p[bh * N * N + i * N + j] /= sum;
        }
    }

    // ===== 3) dV = P^T @ dout =====
    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        D, N, N,
        &alpha,
        p, CUDA_R_32F, N, N * N,
        dout, CUDA_R_32F, D, N * D,
        &beta,
        dV, CUDA_R_32F, D, N * D,
        B * H,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // ===== 4) grad_S = dout @ V^T =====
    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        N, N, D,
        &alpha,
        dout, CUDA_R_32F, D, N * D,
        v, CUDA_R_32F, D, N * D,
        &beta,
        grad_s, CUDA_R_32F, N, N * N,
        B * H,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // ===== 5) softmax 反向 =====
    for (int bh = 0; bh < B * H; ++bh) {
        for (int i = 0; i < N; ++i) {
            float* row_grad = grad_s + bh * N * N + i * N;
            const float* row_out = p + bh * N * N + i * N;
            softmax_backward_row_device<<<(N + 127) / 128, 128, 128 * sizeof(float)>>>(row_grad, row_out, N);
        }
    }
    cudaDeviceSynchronize();

    // ===== 6) dQ = grad_S @ K =====
    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        D, N, N,
        &alpha,
        k, CUDA_R_32F, D, N * D,
        grad_s, CUDA_R_32F, N, N * N,
        &beta,
        dQ, CUDA_R_32F, D, N * D,
        B * H,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // ===== 7) dK = grad_S^T @ Q =====
    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        D, N, N,
        &alpha,
        q, CUDA_R_32F, D, N * D,
        grad_s, CUDA_R_32F, N, N * N,
        &beta,
        dK, CUDA_R_32F, D, N * D,
        B * H,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cudaFree(s); cudaFree(p); cudaFree(grad_s);
    return {dQ, dK, dV};
}