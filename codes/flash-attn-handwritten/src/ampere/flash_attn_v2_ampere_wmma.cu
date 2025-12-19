#include "ampere/flash_attn_v2_ampere_wmma.h"
#include <cuda_runtime.h>
#include <mma.h>          // WMMA
#include <cstdio>

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

using namespace nvcuda;

__global__ void flash_v2_ampere_wmma_kernel(
    const float* q, const float* k, const float* v, float* out,
    int N, int D)
{
    extern __shared__ float smem[];          // 2*ND + NN
    float* sq = smem;                        // Q tile [M,K]
    float* sk = smem + WMMA_M * D;           // K tile [N,K]
    float* sv = smem + 2 * WMMA_M * D;       // V tile [N,D]
    float* ss = smem + 2 * WMMA_M * D + WMMA_N * D; // S [M,N] 临时

    int warpM = blockIdx.y * WMMA_M;
    int warpN = blockIdx.z * WMMA_N;
    if (warpM >= N || warpN >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_q;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_k;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_s;

    // 1) 加载 Q tile 到共享内存
    for (int i = threadIdx.x; i < WMMA_M * D; i += blockDim.x)
        sq[i] = q[warpM * D + i];
    __syncthreads();

    // 2) 加载 K tile
    for (int i = threadIdx.x; i < WMMA_N * D; i += blockDim.x)
        sk[i] = k[warpN * D + i];
    __syncthreads();

    // 3) S = Q K^T   (WMMA)
    wmma::fill_fragment(frag_s, 0.0f);
    for (int kTile = 0; kTile < D; kTile += WMMA_K) {
        wmma::load_matrix_sync(frag_q, (const half*)sq, D);
        wmma::load_matrix_sync(frag_k, (const half*)sk, D);
        wmma::mma_sync(frag_s, frag_q, frag_k, frag_s);
    }
    wmma::store_matrix_sync(ss, frag_s, WMMA_N, wmma::mem_row_major);

    // 4) row-wise softmax
    for (int row = threadIdx.x; row < WMMA_M; row += blockDim.x) {
        float maxVal = -1e38f;
        for (int col = 0; col < WMMA_N; ++col)
            maxVal = fmaxf(maxVal, ss[row * WMMA_N + col]);
        float sum = 0.0f;
        for (int col = 0; col < WMMA_N; ++col) {
            float ev = expf(ss[row * WMMA_N + col] - maxVal);
            ss[row * WMMA_N + col] = ev;
            sum += ev;
        }
        for (int col = 0; col < WMMA_N; ++col)
            ss[row * WMMA_N + col] /= sum;
    }
    __syncthreads();

    // 5) 加载 V tile
    for (int i = threadIdx.x; i < WMMA_N * D; i += blockDim.x)
        sv[i] = v[warpN * D + i];
    __syncthreads();

    // 6) O = S V   (再次 WMMA)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_s2;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_v;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_o;
    wmma::fill_fragment(frag_o, 0.0f);
    for (int kTile = 0; kTile < D; kTile += WMMA_K) {
        wmma::load_matrix_sync(frag_s2, (const half*)ss, WMMA_N);
        wmma::load_matrix_sync(frag_v, (const half*)sv, D);
        wmma::mma_sync(frag_o, frag_s2, frag_v, frag_o);
    }
    // 写回全局
    wmma::store_matrix_sync(out + warpM * D, frag_o, D, wmma::mem_row_major);
}

void flash_attn_v2_ampere_wmma_fwd(const float* q, const float* k, const float* v,
                            float* out,
                            int B, int H, int N, int D)
{
    dim3 block(128);
    dim3 grid(1, (N + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);
    size_t smem = (2 * WMMA_M * D + WMMA_N * D + WMMA_M * WMMA_N) * sizeof(float);
    for (int bh = 0; bh < B * H; ++bh)
        flash_v2_ampere_wmma_kernel<<<grid, block, smem>>>(
            q + bh * N * D, k + bh * N * D, v + bh * N * D, out + bh * N * D,
            N, D);
    cudaDeviceSynchronize();
}