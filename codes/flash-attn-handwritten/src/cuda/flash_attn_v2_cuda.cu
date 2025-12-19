#include "cuda/flash_attn_v2_cuda.cuh"
#include <cuda_runtime.h>
#include <stdio.h>  // 传统 C 头文件
#define CHECK_CUDA(x)           \
    do                          \
    {                           \
        if ((x) != cudaSuccess) \
            __builtin_trap();   \
    } while (0)

__global__ void flash_attn_v2_cuda_kernel(
    const float *q, const float *k, const float *v, float *out,
    int N, int D)
{
    // 实现与上次相同，仅 kernel 名带 v2
    extern __shared__ float smem[]; // 大小 = 2*N*D*sizeof(float)
    float *s = smem;                // [N,N]  临时矩阵
    float *o = smem + N * N;        // [N,D]  结果

    int tid = threadIdx.x;
    int stride = N * D;

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
    printf("func=%s,line=%d\n", __func__, __LINE__);
    size_t smem = (N * N + N * D) * sizeof(float);
    dim3 grid(B * H, 1, 1);
    dim3 block(128, 1, 1);
    printf("func=%s,line=%d\n", __func__, __LINE__);
    for (int bh = 0; bh < B * H; ++bh)
    {
        flash_attn_v2_cuda_kernel<<<grid, block, smem>>>(
            q + bh * N * D, k + bh * N * D, v + bh * N * D, out + bh * N * D,
            N, D);
    }
    printf("func=%s,line=%d\n", __func__, __LINE__);
    CHECK_CUDA(cudaGetLastError());
        printf("func=%s,line=%d\n", __func__, __LINE__);
    CHECK_CUDA(cudaDeviceSynchronize());
        printf("func=%s,line=%d\n", __func__, __LINE__);
}