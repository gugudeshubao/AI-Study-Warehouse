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




constexpr int BLOCK = 128;

// ----------- 设备端辅助 -----------
__device__ void warp_softmax_backward(float* grad, const float* out, int n) {
    float sum = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x)
        sum += grad[j] * out[j];
    __shared__ float shared_sum;
    __syncthreads();
    if (threadIdx.x == 0) shared_sum = sum;
    __syncthreads();
    for (int j = threadIdx.x; j < n; j += blockDim.x)
        grad[j] = out[j] * (grad[j] - shared_sum);
}

__global__ void flash_v2_cuda_backward_kernel(
    const float* q, const float* k, const float* v,
    const float* out, const float* dout,
    float* dQ, float* dK, float* dV,
    int N, int D)
{
    extern __shared__ float smem[];
    float* s   = smem;                       // [N,N]
    float* p   = s + N*N;
    float* gs  = p + N*N;
    float* tmp = gs + N*N;                   // 临时

    int tid = threadIdx.x;
    int bid = blockIdx.x;   // 1 block per (B,H)
    int stride = N * D;
    q += bid * stride; k += bid * stride; v += bid * stride;
    out += bid * stride; dout += bid * stride;
    dQ += bid * stride; dK += bid * stride; dV += bid * stride;

    // 1) S = QK^T
    for (int idx = tid; idx < N * N; idx += blockDim.x) {
        int i = idx / N, j = idx % N;
        float sum = 0.0f;
        for (int d = 0; d < D; ++d)
            sum += q[i * D + d] * k[j * D + d];
        s[idx] = sum;
    }
    __syncthreads();

    // 2) P = softmax(S)
    for (int i = 0; i < N; ++i) {
        // max
        float maxVal = -1e38f;
        for (int j = tid; j < N; j += blockDim.x)
            maxVal = fmaxf(maxVal, s[i * N + j]);
        __shared__ float shared_max;
        __syncthreads();
        if (tid == 0) shared_max = maxVal;
        __syncthreads();
        // exp & sum
        float sum = 0.0f;
        for (int j = tid; j < N; j += blockDim.x) {
            float ev = expf(s[i * N + j] - shared_max);
            p[i * N + j] = ev;
            sum += ev;
        }
        __syncthreads();
        if (tid == 0) shared_max = sum;   // 复用变量
        __syncthreads();
        // normalize
        for (int j = tid; j < N; j += blockDim.x)
            p[i * N + j] /= shared_max;
        __syncthreads();
    }

    // 3) dV = P^T @ dout
    for (int idx = tid; idx < N * D; idx += blockDim.x) {
        int j = idx / D, d = idx % D;
        float sum = 0.0f;
        for (int i = 0; i < N; ++i)
            sum += p[i * N + j] * dout[i * D + d];
        dV[idx] = sum;
    }
    __syncthreads();

    // 4) grad_S = dout @ V^T
    for (int idx = tid; idx < N * N; idx += blockDim.x) {
        int i = idx / N, j = idx % N;
        float sum = 0.0f;
        for (int d = 0; d < D; ++d)
            sum += dout[i * D + d] * v[j * D + d];
        gs[idx] = sum;
    }
    __syncthreads();

    // 5) softmax 反向
    for (int i = 0; i < N; ++i) {
        if (tid < N) warp_softmax_backward(&gs[i * N], &p[i * N], N);
        __syncthreads();
    }

    // 6) dQ = grad_S @ K
    for (int idx = tid; idx < N * D; idx += blockDim.x) {
        int i = idx / D, d = idx % D;
        float sum = 0.0f;
        for (int j = 0; j < N; ++j)
            sum += gs[i * N + j] * k[j * D + d];
        dQ[idx] = sum;
    }
    __syncthreads();

    // 7) dK = grad_S^T @ Q
    for (int idx = tid; idx < N * D; idx += blockDim.x) {
        int j = idx / D, d = idx % D;
        float sum = 0.0f;
        for (int i = 0; i < N; ++i)
            sum += gs[i * N + j] * q[i * D + d];
        dK[idx] = sum;
    }
}

FlashAttnGrad flash_attn_v2_cuda_backward(const float* q, const float* k, const float* v,
                                          const float* out, const float* dout,
                                          int B, int H, int N, int D)
{
    size_t len = B * H * N * D;
    float *dQ, *dK, *dV;
    cudaMalloc(&dQ, len * sizeof(float));
    cudaMalloc(&dK, len * sizeof(float));
    cudaMalloc(&dV, len * sizeof(float));

    size_t smem = (3 * N * N + N * D) * sizeof(float);
    dim3 grid(B * H, 1, 1);
    dim3 block(BLOCK, 1, 1);
    flash_v2_cuda_backward_kernel<<<grid, block, smem>>>(
        q, k, v, out, dout, dQ, dK, dV, N, D);
    cudaDeviceSynchronize();
    return {dQ, dK, dV};
}