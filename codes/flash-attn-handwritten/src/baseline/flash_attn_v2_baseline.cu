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
static __global__ void softmax_inplace_dev(float* s, int total_rows, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;
    float* rowPtr = s + static_cast<size_t>(row) * N;

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
    int total_rows = B * H * N;
    dim3 block(128);
    dim3 grid((total_rows + block.x - 1) / block.x);
    softmax_inplace_dev<<<grid, block>>>(s, total_rows, N);
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
    extern __shared__ float shared[];
    float* s_sum = shared;
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        local_sum += grad[i] * out[i];
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();
    // 归约求和
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        __syncthreads();
    }
    float sum = s_sum[0];
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        grad[i] = out[i] * (grad[i] - sum);
}

FlashAttnGrad cublas_flash_v2_backward(const float* q, const float* k, const float* v,
                                       const float* out, const float* dout,
                                       int B, int H, int N, int D)
{
    size_t len = B * H * N * D;
    float *dQ, *dK, *dV;
    CHECK_CUDA(cudaMalloc(&dQ, len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dK, len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dV, len * sizeof(float)));
    CHECK_CUDA(cudaMemset(dQ, 0, len * sizeof(float)));
    CHECK_CUDA(cudaMemset(dK, 0, len * sizeof(float)));
    CHECK_CUDA(cudaMemset(dV, 0, len * sizeof(float)));

    cublasHandle_t handle = get_cublas_handle();
    const float alpha = 1.0f, beta = 0.0f;
    constexpr auto compute_type = CUBLAS_COMPUTE_32F;
    constexpr auto gemm_algo = CUBLAS_GEMM_DEFAULT;

    // 临时设备缓冲区
    float *s, *grad_s;
    CHECK_CUDA(cudaMalloc(&s, B * H * N * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&grad_s, B * H * N * N * sizeof(float)));
    float *p = s;

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
        compute_type,
        gemm_algo);

    // ===== 2) P = softmax(S) =====
    int total_rows = B * H * N;
    dim3 softmax_block(128);
    dim3 softmax_grid((total_rows + softmax_block.x - 1) / softmax_block.x);
    softmax_inplace_dev<<<softmax_grid, softmax_block>>>(p, total_rows, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // ===== 3) dV = P^T @ dout =====
    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        D, N, N,
        &alpha,
        dout, CUDA_R_32F, D, N * D,
        p, CUDA_R_32F, N, N * N,
        &beta,
        dV, CUDA_R_32F, D, N * D,
        B * H,
        compute_type,
        gemm_algo);

    // ===== 4) grad_S = dout @ V^T =====
    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, N, D,
        &alpha,
        v, CUDA_R_32F, D, N * D,
        dout, CUDA_R_32F, D, N * D,
        &beta,
        grad_s, CUDA_R_32F, N, N * N,
        B * H,
        compute_type,
        gemm_algo);

    // ===== 5) softmax 反向 =====
    constexpr int softmax_bwd_threads = 128;
    for (int row = 0; row < total_rows; ++row) {
        float* row_grad = grad_s + static_cast<size_t>(row) * N;
        const float* row_out = p + static_cast<size_t>(row) * N;
        softmax_backward_row_device<<<1, softmax_bwd_threads, softmax_bwd_threads * sizeof(float)>>>(row_grad, row_out, N);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

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
        compute_type,
        gemm_algo);

    // ===== 7) dK = grad_S^T @ Q =====
    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        D, N, N,
        &alpha,
        q, CUDA_R_32F, D, N * D,
        grad_s, CUDA_R_32F, N, N * N,
        &beta,
        dK, CUDA_R_32F, D, N * D,
        B * H,
        compute_type,
        gemm_algo);

    cudaFree(s);
    cudaFree(grad_s);
    return {dQ, dK, dV};
}
