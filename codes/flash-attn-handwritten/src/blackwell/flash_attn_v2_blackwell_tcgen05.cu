#include "blackwell/flash_attn_v2_blackwell_tcgen05.h"
#include <cuda.h>
#include <cuda_pipeline.h>
#include <mma.h>

using fp8_t = __fp8_e4m3;

constexpr int WARPGROUP_SIZE = 128;
constexpr int M_TILE         = 256;
constexpr int N_TILE         = 128;
constexpr int K_TILE         = 32;   // FP8 累加深度

extern "C" __global__ void flash_v2_blackwell_tcgen05_kernel(
    const fp8_t* __restrict__ q,
    const fp8_t* __restrict__ k,
    const fp8_t* __restrict__ v,
    float* __restrict__ o,
    int N, int D)
{
    // TMEM 双缓冲：O 累加 + KV 输入
    extern __shared__ char tmem[];
    float* tmem_o  = reinterpret_cast<float*>(tmem);                      // [M_TILE, N_TILE]
    fp8_t* tmem_kv = reinterpret_cast<fp8_t*>(tmem_o + M_TILE * N_TILE); // 连续放 K/V

    int tid   = threadIdx.x;
    int tileM = blockIdx.y * M_TILE;
    int tileN = blockIdx.z * N_TILE;

    // 寄存器累加器：256×128 输出，128 thread 覆盖
    float accum[8][4] = {};   // 8×4 = 32 元素 / thread

    // 1) TMA bulk 加载 Q/K/V → TMEM
    if (tid == 0) {
        __cudaptx_tma_load_bulk_tensor(tmem_kv, q_tma_map, tileM, 0,
                                       M_TILE * K_TILE * sizeof(fp8_t));
        __cudaptx_tma_load_bulk_tensor(tmem_kv + M_TILE * K_TILE, k_tma_map, tileN, 0,
                                       N_TILE * K_TILE * sizeof(fp8_t));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // 2) tcgen05.mma.sp 计算 Q@K^T
    for (int k0 = 0; k0 < D; k0 += K_TILE) {
        asm volatile("tcgen05.mma.sp.sync.aligned.m256n128k32.row.col.f32.fp8.fp8.fp32 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7}, "
                     "[%8], [%9], [%10], "
                     "{%11,%12,%13,%14,%15,%16,%17,%18};\n"
                     : "+f"(accum[0][0]), "+f"(accum[0][1]), "+f"(accum[0][2]), "+f"(accum[0][3]),
                       "+f"(accum[1][0]), "+f"(accum[1][1]), "+f"(accum[1][2]), "+f"(accum[1][3])
                     : "l"(tmem_kv),                                    // Q
                       "l"(tmem_kv + M_TILE * K_TILE),                  // K
                       "l"(nullptr),                                    // 稠密模式
                       "f"(accum[0][0]), "f"(accum[0][1]), "f"(accum[0][2]), "f"(accum[0][3]),
                       "f"(accum[1][0]), "f"(accum[1][1]), "f"(accum[1][2]), "f"(accum[1][3]));
    }

    // 3) online-softmax（同 Hopper，256 行）
    __shared__ float smem_max[M_TILE], smem_sum[M_TILE];
    for (int row = tid; row < M_TILE; row += blockDim.x) {
        float maxVal = -1e38f, sum = 0.0f;
        #pragma unroll
        for (int c = 0; c < 4; ++c) maxVal = fmaxf(maxVal, accum[row/16][c]);
        smem_max[row] = maxVal;
        __syncthreads();
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            float ev = expf(accum[row/16][c] - maxVal);
            accum[row/16][c] = ev;
            sum += ev;
        }
        smem_sum[row] = sum;
        __syncthreads();
        #pragma unroll
        for (int c = 0; c < 4; ++c) accum[row/16][c] /= sum;
    }

    // 4) TMA bulk 加载 V → TMEM
    if (tid == 0) {
        __cudaptx_tma_load_bulk_tensor(tmem_kv + 2 * M_TILE * K_TILE, v_tma_map, tileN, 0,
                                       N_TILE * D * sizeof(fp8_t));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // 5) tcgen05.mma.sp 计算 S@V
    asm volatile("tcgen05.mma.sp.sync.aligned.m256n128k32.row.col.f32.fp8.fp8.fp32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, "
                 "[%8], [%9], [%10], "
                 "{%11,%12,%13,%14,%15,%16,%17,%18};\n"
                 : "+f"(accum[0][0]), "+f"(accum[0][1]), "+f"(accum[0][2]), "+f"(accum[0][3]),
                   "+f"(accum[1][0]), "+f"(accum[1][1]), "+f"(accum[1][2]), "+f"(accum[1][3])
                 : "l"(tmem_o),                                         // S
                   "l"(tmem_kv + 2 * M_TILE * K_TILE),                  // V
                   "l"(nullptr),
                   "f"(accum[0][0]), "f"(accum[0][1]), "f"(accum[0][2]), "f"(accum[0][3]),
                   "f"(accum[1][0]), "f"(accum[1][1]), "f"(accum[1][2]), "f"(accum[1][3]));

    // 6) 写回全局 O
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 4; ++j) {
            int gRow = tileM + (tid / 16) * 8 + i;
            int gCol = tileN + (tid & 15) * 4 + j;
            if (gRow < N && gCol < D)
                o[gRow * D + gCol] = accum[i][j];
        }
}

// -------------- host wrapper --------------
void flash_attn_v2_blackwell_tcgen05_fwd(const fp8_t* q, const fp8_t* k, const fp8_t* v,
                                         float* o,
                                         int B, int H, int N, int D)
{
    dim3 block(WARPGROUP_SIZE, 1, 1);
    dim3 grid(1, (N + M_TILE - 1) / M_TILE, (N + N_TILE - 1) / N_TILE);
    size_t tmemBytes = (M_TILE * N_TILE * sizeof(float) +
                        (M_TILE + N_TILE + N_TILE) * K_TILE * sizeof(fp8_t));
    for (int bh = 0; bh < B * H; ++bh)
        flash_v2_blackwell_tcgen05_kernel<<<grid, block, tmemBytes>>>(
            q + bh * N * D, k + bh * N * D, v + bh * N * D, o + bh * N * D,
            N, D);
    cudaDeviceSynchronize();
}