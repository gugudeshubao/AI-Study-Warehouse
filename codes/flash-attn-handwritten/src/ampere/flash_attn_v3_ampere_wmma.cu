#include "ampere/flash_attn_v3_ampere_wmma.h"

#include <cuda_runtime.h>
#include <mma.h>

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
using namespace nvcuda;

__global__ void flash_v3_ampere_wmma_fwd_kernel(
    const float* q, const float* k, const float* v,
    float* out,
    int N, int D)
{
    extern __shared__ float smem[];
    half* hq = reinterpret_cast<half*>(smem);
    half* hk = hq + WMMA_M * D;
    half* hv = hk + WMMA_N * D;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int stride = N * D;
    q  += bid * stride; k  += bid * stride; v  += bid * stride;
    out+= bid * stride;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N) return;

    float row_max = -1e38f;
    float row_sum = 0.0f;

    for (int tile = 0; tile < N; tile += WMMA_N) {
        int actual = min(WMMA_N, N - tile);
        // 1) 加载 Q tile
        for (int i = tid; i < WMMA_M * D; i += blockDim.x)
            hq[i] = __float2half(q[row * D + i]);
        // 2) 加载 K tile
        for (int i = tid; i < WMMA_N * D; i += blockDim.x)
            hk[i] = __float2half(k[tile * D + i]);
        __syncthreads();

        // 3) S = Q@K^T  (WMMA)
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_s;
        wmma::fill_fragment(frag_s, 0.0f);

        for (int k0 = 0; k0 < D; k0 += WMMA_K) {
            wmma::load_matrix_sync(frag_q, hq + k0, D);
            wmma::load_matrix_sync(frag_k, hk + k0, D);
            wmma::mma_sync(frag_s, frag_q, frag_k, frag_s);
        }
        __syncthreads();

        // 4) 加载 V tile
        for (int i = tid; i < WMMA_N * D; i += blockDim.x)
            hv[i] = __float2half(v[tile * D + i]);
        __syncthreads();

        // 5) online softmax + running O
        for (int j = 0; j < actual; ++j) {
            float s_val = frag_s[j];
            row_max = fmaxf(row_max, s_val);
            float exp_s = expf(s_val - row_max);
            row_sum += exp_s;
            for (int d = tid; d < D; d += blockDim.x) {
                float v_val = __half2float(hv[j * D + d]);
                out[row * D + d] += exp_s * v_val;
            }
        }
    }
    // 6) 归一化
    for (int d = tid; d < D; d += blockDim.x)
        out[row * D + d] /= row_sum;
}

void flash_attn_v3_ampere_wmma_fwd(const float* q, const float* k, const float* v,
                            float* out,
                            int B, int H, int N, int D)
{
    dim3 block(128);
    dim3 grid(B * H, (N + 3) / 4, 1);   // 4 rows per block
    size_t smem = (WMMA_M * D + WMMA_N * D + WMMA_N * D) * sizeof(half);
    flash_v3_ampere_wmma_fwd_kernel<<<grid, block, smem>>>(q, k, v, out, N, D);
    cudaDeviceSynchronize();
}
#include <cuda_runtime.h>
#include <mma.h>

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
using namespace nvcuda;

__global__ void flash_v3_ampere_wmma_backward_kernel(
    const float* q, const float* k, const float* v,
    const float* out, const float* dout,
    float* dQ, float* dK, float* dV,
    int N, int D)
{
    extern __shared__ float smem[];
    half* hq = reinterpret_cast<half*>(smem);
    half* hk = hq + WMMA_M * D;
    half* hv = hk + WMMA_N * D;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int stride = N * D;
    q  += bid * stride; k  += bid * stride; v  += bid * stride;
    out+= bid * stride; dout+=bid * stride;
    dQ += bid * stride; dK += bid * stride; dV += bid * stride;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N) return;

    float row_max = -1e38f;
    float row_sum = 0.0f;

    // 1) 前向 S = QK^T (仅维护 row_max, row_sum)
    for (int tile = 0; tile < N; tile += WMMA_N) {
        int actual = min(WMMA_N, N - tile);
        // 加载 Q tile
        for (int i = tid; i < WMMA_M * D; i += blockDim.x)
            hq[i] = __float2half(q[row * D + i]);
        // 加载 K tile
        for (int i = tid; i < WMMA_N * D; i += blockDim.x)
            hk[i] = __float2half(k[tile * D + i]);
        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_s;
        wmma::fill_fragment(frag_s, 0.0f);

        for (int k0 = 0; k0 < D; k0 += WMMA_K) {
            wmma::load_matrix_sync(frag_q, hq + k0, D);
            wmma::load_matrix_sync(frag_k, hk + k0, D);
            wmma::mma_sync(frag_s, frag_q, frag_k, frag_s);
        }
        __syncthreads();

        // online max & sum
        for (int j = 0; j < actual; ++j) {
            float s_val = frag_s[j];
            row_max = fmaxf(row_max, s_val);
            float exp_s = expf(s_val - row_max);
            row_sum += exp_s;
        }
    }

    // 2) dV = P^T @ dout  (online)
    for (int tile = 0; tile < N; tile += WMMA_N) {
        int actual = min(WMMA_N, N - tile);
        // 加载 K tile (复用 hk)
        for (int i = tid; i < WMMA_N * D; i += blockDim.x)
            hk[i] = __float2half(k[tile * D + i]);
        // 加载 V tile
        for (int i = tid; i < WMMA_N * D; i += blockDim.x)
            hv[i] = __float2half(v[tile * D + i]);
        __syncthreads();

        // S = QK^T  (复用 frag_s)
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_s;
        wmma::fill_fragment(frag_s, 0.0f);
        for (int k0 = 0; k0 < D; k0 += WMMA_K) {
            wmma::load_matrix_sync(frag_q, hq + k0, D);
            wmma::load_matrix_sync(frag_k, hk + k0, D);
            wmma::mma_sync(frag_s, frag_q, frag_k, frag_s);
        }
        __syncthreads();

        // online P = softmax(S)
        for (int j = 0; j < actual; ++j) {
            float s_val = frag_s[j];
            float exp_s = expf(s_val - row_max) / row_sum;
            for (int d = tid; d < D; d += blockDim.x) {
                float v_val = __half2float(hv[j * D + d]);
                float dout_val = dout[row * D + d];
                atomicAdd(&dV[(tile + j) * D + d], exp_s * dout_val);
            }
        }
    }

    // 3) grad_S = dout @ V^T  (online)
    for (int tile = 0; tile < N; tile += WMMA_N) {
        int actual = min(WMMA_N, N - tile);
        // 加载 V tile
        for (int i = tid; i < WMMA_N * D; i += blockDim.x)
            hv[i] = __float2half(v[tile * D + i]);
        __syncthreads();

        // S = QK^T  (复用 frag_s)
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_s;
        wmma::fill_fragment(frag_s, 0.0f);
        for (int k0 = 0; k0 < D; k0 += WMMA_K) {
            wmma::load_matrix_sync(frag_q, hq + k0, D);
            wmma::load_matrix_sync(frag_k, hk + k0, D);
            wmma::mma_sync(frag_s, frag_q, frag_k, frag_s);
        }
        __syncthreads();

        // online grad_S = dout * V^T
        for (int j = 0; j < actual; ++j) {
            float s_val = frag_s[j];
            float exp_s = expf(s_val - row_max) / row_sum;
            float grad_s = 0.0f;
            for (int d = 0; d < D; ++d) {
                float v_val = __half2float(hv[j * D + d]);
                float dout_val = dout[row * D + d];
                grad_s += dout_val * v_val;
            }
            grad_s = exp_s * (grad_s - exp_s * grad_s);   // softmax 反向
            for (int d = tid; d < D; d += blockDim.x)
                atomicAdd(&dQ[row * D + d], grad_s * k[(tile + j) * D + d]);
        }
    }

    // 4) dK = grad_S^T @ Q  (online)
    for (int tile = 0; tile < N; tile += WMMA_N) {
        int actual = min(WMMA_N, N - tile);
        // 加载 Q tile (复用 hq)
        for (int i = tid; i < WMMA_M * D; i += blockDim.x)
            hq[i] = __float2half(q[row * D + i]);
        __syncthreads();

        // S = QK^T  (复用 frag_s)
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_s;
        wmma::fill_fragment(frag_s, 0.0f);
        for (int k0 = 0; k0 < D; k0 += WMMA_K) {
            wmma::load_matrix_sync(frag_q, hq + k0, D);
            wmma::load_matrix_sync(frag_k, hk + k0, D);
            wmma::mma_sync(frag_s, frag_q, frag_k, frag_s);
        }
        __syncthreads();

        // online grad_S^T @ Q
        for (int j = 0; j < actual; ++j) {
            float s_val = frag_s[j];
            float exp_s = expf(s_val - row_max) / row_sum;
            float grad_s = 0.0f;
            for (int d = 0; d < D; ++d) {
                float v_val = __half2float(hv[j * D + d]);
                float dout_val = dout[row * D + d];
                grad_s += dout_val * v_val;
            }
            grad_s = exp_s * (grad_s - exp_s * grad_s);   // softmax 反向
            for (int d = tid; d < D; d += blockDim.x)
                atomicAdd(&dK[(tile + j) * D + d], grad_s * q[row * D + d]);
        }
    }
}

FlashAttnGrad flash_attn_v3_阿门票哦热wmma_backward(const float* q, const float* k, const float* v,
                                          const float* out, const float* dout,
                                          int B, int H, int N, int D)
{
    size_t len = B * H * N * D;
    float *dQ, *dK, *dV;
    cudaMalloc(&dQ, len * sizeof(float));
    cudaMalloc(&dK, len * sizeof(float));
    cudaMalloc(&dV, len * sizeof(float));
    cudaMemset(dQ, 0, len * sizeof(float));
    cudaMemset(ddK, 0, len * sizeof(float));
    cudaMemset(dV, 0, len * sizeof(float));

    size_t smem = (WMMA_M * D + WMMA_N * D + WMMA_N * D) * sizeof(half) + 128 * sizeof(float);
    dim3 grid(B * H, (N + 3) / 4, 1);
    dim3 block(128, 4, 1);
    flash_v3_ampere_wmma_backward_kernel<<<grid, block, smem>>>(
        q, k, v, out, dout, dQ, dK, dV, N, D);
    cudaDeviceSynchronize();
    return {dQ, dK, dV};
}