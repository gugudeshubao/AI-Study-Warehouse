#include "ampere/flash_attn_v2_ampere_mma.h"
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda/barrier>
// #include <cuda_pipeline.h>
namespace ampere_mma 
{
using barrier = cuda::barrier<cuda::thread_scope_block>;

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 8;

// 一次 warp 处理 16×8 输出 tile
__device__ void load_q_mma(const float* q, half* smem, int stride, int tileM, int D)
{
    int lane = threadIdx.x & 31;
    int row = lane >> 2;        // 0-7
    int col = (lane & 3) * 2;   // 0,2,4,6
    int gRow = tileM + row;
    if (gRow >= stride) return;
    // 4 线程协作 8×8 子块
    uint32_t addr = __cvta_generic_to_shared(smem + row * 8 + col);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, 8;\n" ::
                 "r"(addr), "l"(&q[gRow * D + col]));
}

__device__ void load_k_mma(const float* k, half* smem, int stride, int tileN, int D)
{
    int lane = threadIdx.x & 31;
    int row = lane >> 2;
    int col = (lane & 3) * 2;
    int gRow = tileN + row;
    if (gRow >= stride) return;
    uint32_t addr = __cvta_generic_to_shared(smem + row * 8 + col);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, 8;\n" ::
                 "r"(addr), "l"(&k[gRow * D + col]));
}

__global__ void flash_v2_ampere_mma_kernel(
    const float* q, const float* k, const float* v, float* out,
    int N, int D)
{
    extern __shared__ half smem[];
    half* sq = smem;                                    // Q tile [16, D]
    half* sk = smem + 16 * D;                           // K tile [8, D]
    half* sv = smem + 16 * D + 8 * D;                   // V tile [8, D]
    float* ss = (float*)(smem + 16 * D + 8 * D + 8 * D); // S [16, 8]

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int tileM = blockIdx.y * MMA_M;
    int tileN = blockIdx.z * MMA_N;

    // 1) cp.async 加载 Q tile
    for (int i = 0; i < D; i += 8)
        load_q_mma(q + tileM * D, sq + i, N, 0, D);
    cp_async_wait_group<0>();
    __syncthreads();

    // 2) 加载 K tile
    for (int i = 0; i < D; i += 8)
        load_k_mma(k + tileN * D, sk + i, N, 0, D);
    cp_async_wait_group<0>();
    __syncthreads();

    // 3) 使用 ldmatrix 载入寄存器 + mma
    uint32_t a[2], b[1];
    float c[4] = {0};
    int tileK = (D + MMA_K - 1) / MMA_K;
    for (int kStep = 0; kStep < tileK; ++kStep) {
        // ldmatrix
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                     :: "r"(a[0]), "r"(a[1]), "l"(sq + kStep * MMA_K));
        asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                     :: "r"(b[0]), "l"(sk + kStep * MMA_K));
        // mma
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(b[0]));
    }
    // 写回共享内存 S
    for (int i = 0; i < 4; ++i)
        ss[lane * 4 + i] = c[i];
    __syncthreads();

    // 4) row softmax
    for (int row = threadIdx.x; row < MMA_M; row += blockDim.x) {
        float maxVal = -1e38f;
        for (int col = 0; col < MMA_N; ++col)
            maxVal = fmaxf(maxVal, ss[row * MMA_N + col]);
        float sum = 0.0f;
        for (int col = 0; col < MMA_N; ++col) {
            float ev = expf(ss[row * MMA_N + col] - maxVal);
            ss[row * MMA_N + col] = ev;
            sum += ev;
        }
        for (int col = 0; col < MMA_N; ++col)
            ss[row * MMA_N + col] /= sum;
    }
    __syncthreads();

    // 5) 加载 V tile
    for (int i = 0; i < D; i += 8)
        load_k_mma(v + tileN * D, sv + i, N, 0, D);   // 复用 load_k_mma
    cp_async_wait_group<0>();
    __syncthreads();

    // 6) O = S V  再次 mma
    for (int kStep = 0; kStep < tileK; ++kStep) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                     :: "r"(a[0]), "r"(a[1]), "l"(ss + kStep * MMA_K));
        asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                     :: "r"(b[0]), "l"(sv + kStep * MMA_K));
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(b[0]));
    }
    // 写回全局
    for (int i = 0; i < 4; ++i)
        out[tileM * MMA_N + lane * 4 + i] = c[i];
}
}

void flash_attn_v2_ampere_mma_fwd(const float* q, const float* k, const float* v,
                            float* out,
                            int B, int H, int N, int D)
{
    using namespace ampere_mma;
    dim3 block(128);
    dim3 grid(1, (N + MMA_M - 1) / MMA_M, (N + MMA_N - 1) / MMA_N);
    size_t smem = (16 * D + 8 * D + 8 * D + 16 * 8) * sizeof(half);
    for (int bh = 0; bh < B * H; ++bh)
        flash_v2_ampere_mma_kernel<<<grid, block, smem>>>(
            q + bh * N * D, k + bh * N * D, v + bh * N * D, out + bh * N * D,
            N, D);
    cudaDeviceSynchronize();
}