#include "cuda/flash_attn_v2_cuda.cuh"
#include <cuda_runtime.h>
#include <stdio.h> // 传统 C 头文件


__global__ void flash_attn_v2_cuda_kernel(
    const float *q, const float *k, const float *v, float *out,
    int N, int D)
{

    // 实现与上次相同，仅 kernel 名带 v2
    extern __shared__ float smem[]; // 大小 = 2*N*D*sizeof(float)
    float *s = smem;                // [N,N]  临时矩阵
    float *o = smem + N * N;        // [N,D]  结果

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int stride = N * D;
    q += bid * stride;
    k += bid * stride;
    v += bid * stride;
    out += bid * stride;

    // 1) S = QK^T  每个线程负责一行
    for (int i = tid; i < N; i += blockDim.x)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int d = 0; d < D; ++d)
                sum += q[i * D + d] * k[j * D + d];
            s[i * N + j] = sum;
        }
    }
    __syncthreads();

    // 2) row max
    for (int i = tid; i < N; i += blockDim.x)
    {
        float max_val = -1e38f;
        for (int j = 0; j < N; ++j)
            max_val = fmaxf(max_val, s[i * N + j]);
        // 就地减去 max
        for (int j = 0; j < N; ++j)
            s[i * N + j] = expf(s[i * N + j] - max_val);
    }
    __syncthreads();

    // 3) row sum
    for (int i = tid; i < N; i += blockDim.x)
    {
        float sum_exp = 0.0f;
        for (int j = 0; j < N; ++j)
            sum_exp += s[i * N + j];
        for (int j = 0; j < N; ++j)
            s[i * N + j] /= sum_exp;
    }
    __syncthreads();

    // 4) O = S V
    for (int i = tid; i < N; i += blockDim.x)
    {
        for (int d = 0; d < D; ++d)
        {
            float acc = 0.0f;
            for (int j = 0; j < N; ++j)
                acc += s[i * N + j] * v[j * D + d];
            o[i * D + d] = acc;
        }
    }
    __syncthreads();

    // 写回全局
    for (int i = tid; i < N * D; i += blockDim.x)
        out[i] = o[i];
}

void flash_attn_v2_cuda_fwd(const float *q, const float *k, const float *v,
                            float *out,
                            int B, int H, int N, int D)
{

    dim3 grid(B * H, 1, 1);
    dim3 block(128, 1, 1);
    CHECK_CUDA_LAST();
    size_t shm_size = (N*D+N*N) * sizeof(float);
    flash_attn_v2_cuda_kernel<<<grid, block,shm_size>>>(q, k, v, out, N, D);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

constexpr int BLOCK = 128;

__global__ void flash_v2_cuda_backward_kernel(
    const float *q, const float *k, const float *v,
    const float *out, const float *dout,
    float *dQ, float *dK, float *dV,
    int N, int D)
{
    extern __shared__ float smem[];
    float *s = smem; // [N,N]
    float *p = s + N * N;
    float *gs = p + N * N;

    int tid = threadIdx.x;
    int bid = blockIdx.x; // 1 block per (B,H)
    int stride = N * D;
    q += bid * stride;
    k += bid * stride;
    v += bid * stride;
    out += bid * stride;
    dout += bid * stride;
    dQ += bid * stride;
    dK += bid * stride;
    dV += bid * stride;

    // 1) S = QK^T
    for (int idx = tid; idx < N * N; idx += blockDim.x)
    {
        int i = idx / N, j = idx % N;
        float sum = 0.0f;
        for (int d = 0; d < D; ++d)
            sum += q[i * D + d] * k[j * D + d];
        s[idx] = sum;
    }
    __syncthreads();

    // 2) P = softmax(S)
    for (int i = 0; i < N; ++i)
    {
        if (tid == 0)
        {
            float maxVal = -1e38f;
            for (int j = 0; j < N; ++j)
                maxVal = fmaxf(maxVal, s[i * N + j]);

            float sum = 0.0f;
            for (int j = 0; j < N; ++j)
            {
                float ev = expf(s[i * N + j] - maxVal);
                p[i * N + j] = ev;
                sum += ev;
            }

            for (int j = 0; j < N; ++j)
                p[i * N + j] /= sum;
        }
        __syncthreads();
    }

    // 3) dV = P^T @ dout
    for (int idx = tid; idx < N * D; idx += blockDim.x)
    {
        int j = idx / D, d = idx % D;
        float sum = 0.0f;
        for (int i = 0; i < N; ++i)
            sum += p[i * N + j] * dout[i * D + d];
        dV[idx] = sum;
    }
    __syncthreads();

    // 4) grad_S = dout @ V^T
    for (int idx = tid; idx < N * N; idx += blockDim.x)
    {
        int i = idx / N, j = idx % N;
        float sum = 0.0f;
        for (int d = 0; d < D; ++d)
            sum += dout[i * D + d] * v[j * D + d];
        gs[idx] = sum;
    }
    __syncthreads();

    // 5) softmax 反向
    for (int i = 0; i < N; ++i)
    {
        if (tid == 0)
        {
            float sum = 0.0f;
            for (int j = 0; j < N; ++j)
                sum += gs[i * N + j] * p[i * N + j];
            for (int j = 0; j < N; ++j)
                gs[i * N + j] = p[i * N + j] * (gs[i * N + j] - sum);
        }
        __syncthreads();
    }

    // 6) dQ = grad_S @ K
    for (int idx = tid; idx < N * D; idx += blockDim.x)
    {
        int i = idx / D, d = idx % D;
        float sum = 0.0f;
        for (int j = 0; j < N; ++j)
            sum += gs[i * N + j] * k[j * D + d];
        dQ[idx] = sum;
    }
    __syncthreads();

    // 7) dK = grad_S^T @ Q
    for (int idx = tid; idx < N * D; idx += blockDim.x)
    {
        int j = idx / D, d = idx % D;
        float sum = 0.0f;
        for (int i = 0; i < N; ++i)
            sum += gs[i * N + j] * q[i * D + d];
        dK[idx] = sum;
    }
}

FlashAttnGrad flash_attn_v2_cuda_backward(const float *q, const float *k, const float *v,
                                          const float *out, const float *dout,
                                          int B, int H, int N, int D)
{
    size_t len = B * H * N * D;
    float *dQ, *dK, *dV;
    CHECK_CUDA(cudaMalloc(&dQ, len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dK, len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dV, len * sizeof(float)));

    size_t smem = (3 * N * N + N * D) * sizeof(float);
    dim3 grid(B * H, 1, 1);
    dim3 block(BLOCK, 1, 1);
    flash_v2_cuda_backward_kernel<<<grid, block, smem>>>(
        q, k, v, out, dout, dQ, dK, dV, N, D);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    return {dQ, dK, dV};
}
