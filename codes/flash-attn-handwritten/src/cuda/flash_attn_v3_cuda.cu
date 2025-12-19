#include "cuda/flash_attn_v3_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>

constexpr int TILE_N = 32;
constexpr int BLOCK = 128;

__global__ void flash_v3_cuda_fwd_kernel(
    const float* q, const float* k, const float* v,
    float* out,
    int N, int D)
{
    extern __shared__ float smem[];
    float* tile_p = smem;                       // [TILE_N]
    float* tile_v = smem + TILE_N;              // [TILE_N,D]
    int tid = threadIdx.x;
    int bid = blockIdx.x;   // 1 block per (B,H)
    int stride = N * D;
    q  += bid * stride; k  += bid * stride; v  += bid * stride;
    out+= bid * stride;

    int row = blockIdx.y * blockDim.y + threadIdx.y;   // 1 thread per row
    if (row >= N) return;

    float row_max = -1e38f;
    float row_sum = 0.0f;
    for (int tile = 0; tile < N; tile += TILE_N) {
        int actual = min(TILE_N, N - tile);
        // 1) 加载 tile_p = S[row][tile:tile+actual]
        for (int jj = tid; jj < actual; jj += blockDim.x) {
            int j = tile + jj;
            float dot = 0.0f;
            for (int d = 0; d < D; ++d)
                dot += q[row * D + d] * k[j * D + d];
            tile_p[jj] = dot;
            if (dot > row_max) row_max = dot;
        }
        __syncthreads();
        // 2) 加载 tile_v = V[tile:tile+actual][d]
        for (int d = 0; d < D; ++d) {
            for (int jj = tid; jj < actual; jj += blockDim.x)
                tile_v[jj * D + d] = v[(tile + jj) * D + d];
        }
        __syncthreads();
        // 3) online softmax + running O
        for (int jj = 0; jj < actual; ++jj) {
            float p = expf(tile_p[jj] - row_max);
            row_sum += p;
            for (int d = 0; d < D; ++d)
                out[row * D + d] += p * tile_v[jj * D + d];
        }
        __syncthreads();
    }
    // 4) 归一化
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        out[row * D + d] /= row_sum;
}

void flash_attn_v3_cuda_fwd(const float* q, const float* k, const float* v,
                            float* out,
                            int B, int H, int N, int D)
{
    dim3 block(128);
    dim3 grid(B * H, (N + 3) / 4, 1);   // 4 rows per block
    size_t smem = (TILE_N + TILE_N * D) * sizeof(float);
    flash_v3_cuda_fwd_kernel<<<grid, block, smem>>>(q, k, v, out, N, D);
    cudaDeviceSynchronize();
}


__device__ void warp_softmax_backward(float* grad, const float* out, int n) {
    extern __shared__ float shared[];
    float* s_sum = shared;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sum += grad[i] * out[i];
    s_sum[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) sum = s_sum[0];
    __syncthreads();
    int idx = threadIdx.x;
    if (idx < n) grad[idx] = out[idx] * (grad[idx] - sum);
}

__global__ void flash_v3_cuda_backward_kernel(
    const float* q, const float* k, const float* v,
    const float* out, const float* dout,
    float* dQ, float* dK, float* dV,
    int N, int D)
{
    extern __shared__ float smem[];
    float* tile_p = smem;                       // [TILE_N]
    float* tile_v = smem + TILE_N;              // [TILE_N,D]
    float* tile_gs = smem + TILE_N + TILE_N * D; // [TILE_N]
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int stride = N * D;
    q  += bid * stride; k  += bid * stride; v  += bid * stride;
    out+= bid * stride; dout+=bid * stride;
    dQ += bid * stride; dK += bid * stride; dV += bid * stride;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N) return;

    // 1) 复用前向：row_max, row_sum
    float row_max = -1e38f;
    float row_sum = 0.0f;
    for (int tile = 0; tile < N; tile += TILE_N) {
        int actual = min(TILE_N, N - tile);
        for (int jj = tid; jj < actual; jj += blockDim.x) {
            int j = tile + jj;
            float dot = 0.0f;
            for (int d = 0; d < D; ++d)
                dot += q[row * D + d] * k[j * D + d];
            tile_p[jj] = dot;
            if (dot > row_max) row_max = dot;
        }
        __syncthreads();
        for (int jj = tid; jj < actual; jj += blockDim.x)
            row_sum += expf(tile_p[jj] - row_max);
        __syncthreads();
    }

    // 2) dV = P^T @ dout  (online)
    for (int tile = 0; tile < N; tile += TILE_N) {
        int actual = min(TILE_N, N - tile);
        // tile_p = exp(S-row_max)/row_sum
        for (int jj = tid; jj < actual; jj += blockDim.x) {
            int j = tile + jj;
            float dot = 0.0f;
            for (int d = 0; d < D; ++d)
                dot += q[row * D + d] * k[j * D + d];
            tile_p[jj] = expf(dot - row_max) / row_sum;
        }
        __syncthreads();
        // load V tile
        for (int d = 0; d < D; ++d) {
            for (int jj = tid; jj < actual; jj += blockDim.x)
                tile_v[jj * D + d] = v[(tile + jj) * D + d];
        }
        __syncthreads();
        // running dV
        for (int jj = 0; jj < actual; ++jj) {
            float p = tile_p[jj];
            for (int d = 0; d < D; ++d)
                atomicAdd(&dV[(tile + jj) * D + d], p * dout[row * D + d]);
        }
        __syncthreads();
    }

    // 3) grad_S = dout @ V^T  (online)
    for (int tile = 0; tile < N; tile += TILE_N) {
        int actual = min(TILE_N, N - tile);
        // tile_p = exp(S-row_max)/row_sum
        for (int jj = tid; jj < actual; jj += blockDim.x) {
            int j = tile + jj;
            float dot = 0.0f;
            for (int d = 0; d < D; ++d)
                dot += q[row * D + d] * k[j * D + d];
            tile_p[jj] = expf(dot - row_max) / row_sum;
        }
        __syncthreads();
        // load V tile
        for (int d = 0; d < D; ++d) {
            for (int jj = tid; jj < actual; jj += blockDim.x)
                tile_v[jj * D + d] = v[(tile + jj) * D + d];
        }
        __syncthreads();
        // running grad_S
        for (int jj = 0; jj < actual; ++jj) {
            float pv = 0.0f;
            for (int d = 0; d < D; ++d)
                pv += dout[row * D + d] * tile_v[jj * D + d];
            tile_gs[jj] = pv;
        }
        __syncthreads();
    }
    // softmax 反向
    warp_softmax_backward(tile_gs, tile_p, N);
    __syncthreads();

    // 4) dQ = grad_S @ K
    for (int tile = 0; tile < N; tile += TILE_N) {
        int actual = min(TILE_N, N - tile);
        // load K tile
        for (int d = 0; d < D; ++d) {
            for (int jj = tid; jj < actual; jj += blockDim.x)
                tile_v[jj * D + d] = k[(tile + jj) * D + d];
        }
        __syncthreads();
        // running dQ
        for (int d = 0; d < D; ++d) {
            float sum = 0.0f;
            for (int jj = 0; jj < actual; ++jj)
                sum += tile_gs[jj] * tile_v[jj * D + d];
            atomicAdd(&dQ[row * D + d], sum);
        }
        __syncthreads();
    }

    // 5) dK = grad_S^T @ Q
    for (int tile = 0; tile < N; tile += TILE_N) {
        int actual = min(TILE_N, N - tile);
        // load Q tile
        for (int d = 0; d < D; ++d) {
            for (int jj = tid; jj < actual; jj += blockDim.x)
                tile_v[jj * D + d] = q[(tile + jj) * D + d];
        }
        __syncthreads();
        // running dK
        for (int jj = 0; jj < actual; ++jj) {
            float gs = tile_gs[jj];
            for (int d = 0; d < D; ++d)
                atomicAdd(&dK[(tile + jj) * D + d], gs * q[row * D + d]);
        }
        __syncthreads();
    }
}

FlashAttnGrad flash_attn_v3_cuda_backward(const float* q, const float* k, const float* v,
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

    size_t smem = (TILE_N + TILE_N * D + TILE_N) * sizeof(float) + 128 * sizeof(float);
    dim3 grid(B * H, (N + 3) / 4, 1);   // 4 rows per block
    dim3 block(128, 4, 1);
    flash_v3_cuda_backward_kernel<<<grid, block, smem>>>(
        q, k, v, out, dout, dQ, dK, dV, N, D);
    cudaDeviceSynchronize();
    return {dQ, dK, dV};
}