#include "hopper/flash_attn_v2_hopper_wgmma.h"
#include <cuda.h>
#include <cuda_pipeline.h>
#include <mma.h>          // for wgmma
#include <cuda/barrier>

using barrier = cuda::barrier<cuda::thread_scope_block>;

constexpr int WARPGROUP_SIZE = 128;
constexpr int M_TILE         = 128;
constexpr int N_TILE         = 128;
constexpr int K_TILE         = 16;

// -------------- 设备端 kernel --------------
extern "C" __global__ void flash_v2_hopper_wgmma_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ o,
    int N, int D)
{
    // 1 个 CTA = 1 个 warpgroup = 128 thread
    const int tid = threadIdx.x;
    const int warpId = tid / 32;
    const int lane   = tid & 31;

    // 共享内存双缓冲
    extern __shared__ char smem[];
    half* smem_q = reinterpret_cast<half*>(smem);
    half* smem_k = smem_q + M_TILE * K_TILE;
    half* smem_v = smem_k + N_TILE * K_TILE;
    float* smem_s = reinterpret_cast<float*>(smem_v + N_TILE * K_TILE); // [M_TILE, N_TILE]

    int tileM = blockIdx.y * M_TILE;
    int tileN = blockIdx.z * N_TILE;

    // 寄存器累加器：每个 thread 负责 4×4 子块
    float accum[4][4] = {};

    // 沿 K 维度滑窗
    for (int k0 = 0; k0 < D; k0 += K_TILE) {
        // ===== TMA bulk 加载 Q tile =====
        if (tid == 0) {
            __cudaptx_tma_load_async_bulk_tensor(
                /*dst*/ smem_q,
                /*tensorMap*/ q_tma_map,   // 由 host 提前初始化
                /*crds*/ tileM, k0,
                /*bytes*/ M_TILE * K_TILE * sizeof(half));
        }
        // ===== TMA bulk 加载 K tile =====
        if (tid == 1) {
            __cudaptx_tma_load_async_bulk_tensor(
                smem_k, k_tma_map, tileN, k0,
                N_TILE * K_TILE * sizeof(half));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        // ===== WGMMA: Q@K^T  (128×128×16) =====
        // 使用 nvcuda::wgmma API（PTX 封装）
        nvcuda::wgmma::load_matrix_sync_a(/*desc*/ smem_q, /*ld*/ K_TILE);
        nvcuda::wgmma::load_matrix_sync_b(/*desc*/ smem_k, /*ld*/ K_TILE);
        nvcuda::wgmma::mma_sync(/*accum*/ accum, /*a*/ smem_q, /*b*/ smem_k,
                                /*scale*/ 1.0f, /*zero*/ false);
        __syncthreads();
    }

    // ===== row-wise online-softmax =====
    __shared__ float smem_max[M_TILE], smem_sum[M_TILE];
    for (int row = tid; row < M_TILE; row += blockDim.x) {
        float maxVal = -1e38f, sum = 0.0f;
        #pragma unroll
        for (int colOff = 0; colOff < N_TILE; colOff += 4) {
            #pragma unroll
            for (int c = 0; c < 4; ++c)
                maxVal = fmaxf(maxVal, accum[row & 3][c]);
        }
        smem_max[row] = maxVal;
        __syncthreads();
        for (int colOff = 0; colOff < N_TILE; colOff += 4) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                float ev = expf(accum[row & 3][c] - maxVal);
                accum[row & 3][c] = ev;
                sum += ev;
            }
        }
        smem_sum[row] = sum;
        __syncthreads();
        // normalize
        for (int colOff = 0; colOff < N_TILE; colOff += 4) {
            #pragma unroll
            for (int c = 0; c < 4; ++c)
                accum[row & 3][c] /= sum;
        }
    }

    // ===== TMA bulk 加载 V tile =====
    if (tid == 0) {
        __cudaptx_tma_load_async_bulk_tensor(
            smem_v, v_tma_map, tileN, 0,
            N_TILE * D * sizeof(half));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // ===== WGMMA: S@V =====
    nvcuda::wgmma::load_matrix_sync_a(/*desc*/ smem_s, /*ld*/ N_TILE);
    nvcuda::wgmma::load_matrix_sync_b(/*desc*/ smem_v, /*ld*/ D);
    nvcuda::wgmma::mma_sync(/*dst*/ accum, /*a*/ smem_s, /*b*/ smem_v,
                            /*scale*/ 1.0f, /*zero*/ true);

    // 写回全局 O
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            int gRow = tileM + (tid / 32) * 4 + i;
            int gCol = tileN + (tid & 3) * 4 + j;
            if (gRow < N && gCol < D)
                o[gRow * D + gCol] = accum[i][j];
        }
}

// -------------- host wrapper --------------
void flash_attn_v2_hopper_wgmma_fwd(const float* q, const float* k, const float* v,
                                    float* out,
                                    int B, int H, int N, int D)
{
    dim3 block(WARPGROUP_SIZE, 1, 1);
    dim3 grid(1, (N + M_TILE - 1) / M_TILE, (N + N_TILE - 1) / N_TILE);
    size_t smem = (M_TILE * K_TILE + N_TILE * K_TILE + N_TILE * D + M_TILE * N_TILE) * sizeof(half);
    for (int bh = 0; bh < B * H; ++bh)
        flash_v2_hopper_wgmma_kernel<<<grid, block, smem>>>(
            q + bh * N * D, k + bh * N * D, v + bh * N * D, out + bh * N * D,
            N, D);
    cudaDeviceSynchronize();
}