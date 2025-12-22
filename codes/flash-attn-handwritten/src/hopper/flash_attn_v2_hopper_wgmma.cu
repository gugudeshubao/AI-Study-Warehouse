#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>

constexpr int WG        = 128;   // 1 warpgroup
constexpr int M_TILE    = 128;
constexpr int N_TILE    = 128;
constexpr int K_TILE    = 16;    // mma.m16n8k8 最小 K
constexpr int MAXN      = 1024;
constexpr int MAXD      = 128;

// -------------------- PTX 宏 --------------------
// 全局内存 → 共享内存  2-D bulk async (Hopper)
#define CP_ASYNC_BULK_TENSOR_2D(dst_smem, src_global, height, width_bytes) \
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                 ".rank1.b32 [%0], [%1], [%2], 0x0, 0x0, 0x0, 0x0;"
                 :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem))),
                    "l"(reinterpret_cast<uint64_t>(src_global)),
                    "r"(static_cast<uint32_t>((height) * (width_bytes))))

// ldmatrix 共享内存 → 矩阵寄存器
#define LDMATRIX_X4(r0, r1, r2, r3, addr)                              \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16           \n" \
                 "    {%0, %1, %2, %3}, [%4];                         \n" \
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)             \
                 : "r"(addr))

// WGMMA 64×64×16
#define WGMMA_M64N64K16_F32(d, desc_a, desc_b, scale)                  \
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.f32.f32  \n" \
                 "    {%0,%1,%2,%3}, %4, %5, %6, %7;                   \n" \
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])     \
                 : "l"(desc_a), "l"(desc_b), "f"(scale), "f"(0.0f))

// -------------------- 前向 kernel --------------------
__global__ void flash_v2_hopper_fwd_kernel(
    const float* __restrict__ gq,
    const float* __restrict__ gk,
    const float* __restrict__ gv,
    float*      __restrict__ go,
    int N, int D)
{
    extern __shared__ char smem_[];
    half* smem_q = reinterpret_cast<half*>(smem_);
    half* smem_k = smem_q + M_TILE * K_TILE;
    half* smem_v = smem_k + N_TILE * K_TILE;
    float* smem_s = reinterpret_cast<float*>(smem_v + N_TILE * D);

    int tileM = blockIdx.y * M_TILE;
    int tileN = blockIdx.z * N_TILE;
    int tid   = threadIdx.x;
    int ty    = tid >> 3;   // 0-15
    int tx    = tid & 7;    // 0-7

    float accum[4] = {0};

    // ===== 1) Q@K^T =====
    for (int k0 = 0; k0 < D; k0 += K_TILE) {
        // 1.1 全局内存 → 共享内存  （每个线程负责 4 行）
        const half* gq_ptr = reinterpret_cast<const half*>(
                                gq + (tileM * D + k0));
        const half* gk_ptr = reinterpret_cast<const half*>(
                                gk + (tileN * D + k0));
        // 高度 128 行，宽度 16×2 byte
        if (tid == 0) {
            CP_ASYNC_BULK_TENSOR_2D(smem_q, gq_ptr, M_TILE, K_TILE * 2);
            CP_ASYNC_BULK_TENSOR_2D(smem_k, gk_ptr, N_TILE, K_TILE * 2);
        }
        __syncthreads();

        // 1.2 ldmatrix → 矩阵寄存器 → wgmma
        for (int m64 = 0; m64 < 2; ++m64) {
            for (int n64 = 0; n64 < 2; ++n64) {
                uint32_t ra0, ra1, ra2, ra3;
                uint32_t smem_q_lane = __cvta_generic_to_shared(
                                        smem_q + m64 * 64 * K_TILE + (ty & 3) * 16);
                LDMATRIX_X4(ra0, ra1, ra2, ra3, smem_q_lane);

                uint32_t rb0, rb1, rb2, rb3;
                uint32_t smem_k_lane = __cvta_generic_to_shared(
                                        smem_k + n64 * 64 * K_TILE + (ty & 3) * 16);
                LDMATRIX_X4(rb0, rb1, rb2, rb3, smem_k_lane);

                uint64_t desc_a = __cvta_generic_to_shared(smem_q + m64 * 64 * K_TILE);
                uint64_t desc_b = __cvta_generic_to_shared(smem_k + n64 * 64 * K_TILE);
                WGMMA_M64N64K16_F32(accum, desc_a, desc_b, 1.0f);
            }
        }
        __syncthreads();
    }

    // ===== 2) online softmax + store S =====
    __shared__ float max_[M_TILE], sum_[M_TILE];
    for (int row = tid; row < M_TILE; row += WG) {
        float m = -FLT_MAX, s = 0;
        for (int col = 0; col < N_TILE; ++col)
            m = fmaxf(m, smem_s[row * N_TILE + col]);
        max_[row] = m;
        for (int col = 0; col < N_TILE; ++col) {
            float ev = expf(smem_s[row * N_TILE + col] - m);
            smem_s[row * N_TILE + col] = ev;
            s += ev;
        }
        sum_[row] = s;
        for (int col = 0; col < N_TILE; ++col)
            smem_s[row * N_TILE + col] /= s;
    }
    __syncthreads();

    // ===== 3) S@V =====
    for (int k0 = 0; k0 < D; k0 += K_TILE) {
        const half* gv_ptr = reinterpret_cast<const half*>(
                                gv + (tileN * D + k0));
        if (tid == 0)
            CP_ASYNC_BULK_TENSOR_2D(smem_v, gv_ptr, N_TILE, K_TILE * 2);
        __syncthreads();

        for (int m64 = 0; m64 < 2; ++m64)
            for (int n64 = 0; n64 < D / 64; ++n64) {
                uint64_t ptrA = __cvta_generic_to_shared(smem_s + m64 * 64 * N_TILE);
                uint64_t ptrB = __cvta_generic_to_shared(smem_v + n64 * 64 * K_TILE);
                WGMMA_M64N64K16_F32(accum, ptrA, ptrB, 1.0f);
            }
        __syncthreads();
    }

    // ===== 4) 写回 O =====
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            int gRow = tileM + (tid >> 3) * 4 + i;
            int gCol = tileN + (tid & 7) * 4 + j;
            if (gRow < N && gCol < D)
                go[gRow * D + gCol] = accum[i];
        }
}

// -------------------- host 接口 --------------------
void flash_attn_v2_hopper_wgmma_fwd(
    const float* q, const float* k, const float* v,
    float* out, int B, int H, int N, int D)
{
    dim3 blk(WG, 1, 1);
    dim3 grd(1, (N + M_TILE - 1) / M_TILE, (N + N_TILE - 1) / N_TILE);
    size_t smem = (M_TILE * K_TILE + N_TILE * K_TILE + N_TILE * D + M_TILE * N_TILE) * 2;
    for (int bh = 0; bh < B * H; ++bh)
        flash_v2_hopper_fwd_kernel<<<grd, blk, smem>>>(
            q + bh * N * D, k + bh * N * D, v + bh * N * D,
            out + bh * N * D, N, D);
    cudaDeviceSynchronize();
}

// -------------------- 反向 kernel --------------------
__global__ void flash_v2_hopper_bwd_kernel(
    const float* __restrict__ gq,
    const float* __restrict__ gk,
    const float* __restrict__ gv,
    const float* __restrict__ gout,
    const float* __restrict__ gdout,
    float* __restrict__ gdq,
    float* __restrict__ gdk,
    float* __restrict__ gdv,
    int N, int D)
{
    extern __shared__ char smem_[];
    half* smem_q = reinterpret_cast<half*>(smem_);
    half* smem_k = smem_q + M_TILE * K_TILE;
    half* smem_v = smem_k + N_TILE * K_TILE;
    float* smem_s  = reinterpret_cast<float*>(smem_v + N_TILE * D);
    float* smem_p  = smem_s  + M_TILE * N_TILE;
    float* smem_gs = smem_p  + M_TILE * N_TILE;

    int tileM = blockIdx.y * M_TILE;
    int tileN = blockIdx.z * N_TILE;
    int tid   = threadIdx.x;
    int ty    = tid >> 3;
    int tx    = tid & 7;
    float accum[4];

    // 1) 前向 S = Q@K^T
    for (int k0 = 0; k0 < D; k0 += K_TILE) {
        const half* gq_ptr = reinterpret_cast<const half*>(gq + (tileM * D + k0));
        const half* gk_ptr = reinterpret_cast<const half*>(gk + (tileN * D + k0));
        if (tid == 0) {
            CP_ASYNC_BULK_TENSOR_2D(smem_q, gq_ptr, M_TILE, K_TILE * 2);
            CP_ASYNC_BULK_TENSOR_2D(smem_k, gk_ptr, N_TILE, K_TILE * 2);
        }
        __syncthreads();
        for (int m64 = 0; m64 < 2; ++m64)
            for (int n64 = 0; n64 < 2; ++n64) {
                uint64_t ptrA = __cvta_generic_to_shared(smem_q + m64 * 64 * K_TILE);
                uint64_t ptrB = __cvta_generic_to_shared(smem_k + n64 * 64 * K_TILE);
                WGMMA_M64N64K16_F32(accum, ptrA, ptrB, 1.0f);
            }
        __syncthreads();
    }
    // store S
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            int gRow = tileM + (tid >> 3) * 4 + i;
            int gCol = tileN + (tid & 7) * 4 + j;
            if (gRow < N && gCol < N)
                smem_s[gRow * N + gCol] = accum[i];
        }
    __syncthreads();

    // 2) P = softmax(S)
    __shared__ float max_[M_TILE], sum_[M_TILE];
    for (int row = tid; row < M_TILE; row += WG) {
        float m = -FLT_MAX, s = 0;
        for (int col = 0; col < N_TILE; ++col)
            m = fmaxf(m, smem_s[row * N_TILE + col]);
        max_[row] = m;
        for (int col = 0; col < N_TILE; ++col) {
            float ev = expf(smem_s[row * N_TILE + col] - m);
            smem_p[row * N_TILE + col] = ev;
            s += ev;
        }
        sum_[row] = s;
        for (int col = 0; col < N_TILE; ++col)
            smem_p[row * N_TILE + col] /= s;
    }
    __syncthreads();

    // 3) dV = P^T @ dout
    for (int k0 = 0; k0 < D; k0 += K_TILE) {
        const half* gv_ptr  = reinterpret_cast<const half*>(gv  + (tileN * D + k0));
        const half* gdout_ptr = reinterpret_cast<const half*>(gdout + (tileM * D + k0));
        if (tid == 0) {
            CP_ASYNC_BULK_TENSOR_2D(smem_v,  gv_ptr,  N_TILE, K_TILE * 2);
            CP_ASYNC_BULK_TENSOR_2D(smem_q, gdout_ptr, M_TILE, K_TILE * 2); // 复用 smem_q 存 dout
        }
        __syncthreads();
        for (int m64 = 0; m64 < 2; ++m64)
            for (int n64 = 0; n64 < D / 64; ++n64) {
                uint64_t ptrA = __cvta_generic_to_shared(smem_p + m64 * 64 * N_TILE);   // P^T
                uint64_t ptrB = __cvta_generic_to_shared(smem_v + n64 * 64 * K_TILE);
                WGMMA_M64N64K16_F32(accum, ptrA, ptrB, 1.0f);
            }
        __syncthreads();
    }
    // store dV
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            int gRow = tileN + (tid >> 3) * 4 + i;
            int gCol = tileM + (tid & 7) * 4 + j;
            if (gRow < N && gCol < D)
                gdv[gRow * D + gCol] = accum[i];
        }
    __syncthreads();

    // 4) grad_S = dout @ V^T
    for (int k0 = 0; k0 < D; k0 += K_TILE) {
        const half* gv_ptr = reinterpret_cast<const half*>(gv + (tileN * D + k0));
        if (tid == 0)
            CP_ASYNC_BULK_TENSOR_2D(smem_v, gv_ptr, N_TILE, K_TILE * 2);
        __syncthreads();
        for (int m64 = 0; m64 < 2; ++m64)
            for (int n64 = 0; n64 < D / 64; ++n64) {
                uint64_t ptrA = __cvta_generic_to_shared(smem_q + m64 * 64 * K_TILE); // dout
                uint64_t ptrB = __cvta_generic_to_shared(smem_v + n64 * 64 * K_TILE);
                WGMMA_M64N64K16_F32(accum, ptrA, ptrB, 1.0f);
            }
        __syncthreads();
    }
    // store grad_S
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            int gRow = tileM + (tid >> 3) * 4 + i;
            int gCol = tileN + (tid & 7) * 4 + j;
            if (gRow < N && gCol < N)
                smem_gs[gRow * N + gCol] = accum[i];
        }
    __syncthreads();

    // 5) softmax_bwd
    for (int row = tid; row < M_TILE; row += WG) {
        float tmp = 0;
        for (int col = 0; col < N_TILE; ++col)
            tmp += smem_gs[row * N_TILE + col] * smem_p[row * N_TILE + col];
        for (int col = 0; col < N_TILE; ++col)
            smem_gs[row * N_TILE + col] =
                smem_p[row * N_TILE + col] * (smem_gs[row * N_TILE + col] - tmp);
    }
    __syncthreads();

    // 6) dQ = grad_S @ K
    for (int k0 = 0; k0 < D; k0 += K_TILE) {
        const half* gk_ptr = reinterpret_cast<const half*>(gk + (tileN * D + k0));
        if (tid == 0)
            CP_ASYNC_BULK_TENSOR_2D(smem_k, gk_ptr, N_TILE, K_TILE * 2);
        __syncthreads();
        for (int m64 = 0; m64 < 2; ++m64)
            for (int n64 = 0; n64 < D / 64; ++n64) {
                uint64_t ptrA = __cvta_generic_to_shared(smem_gs + m64 * 64 * N_TILE);
                uint64_t ptrB = __cvta_generic_to_shared(smem_k + n64 * 64 * K_TILE);
                WGMMA_M64N64K16_F32(accum, ptrA, ptrB, 1.0f);
            }
        __syncthreads();
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            int gRow = tileM + (tid >> 3) * 4 + i;
            int gCol = tileN + (tid & 7) * 4 + j;
            if (gRow < N && gCol < D)
                gdq[gRow * D + gCol] = accum[i];
        }
    __syncthreads();

    // 7) dK = grad_S^T @ Q
    for (int k0 = 0; k0 < D; k0 += K_TILE) {
        const half* gq_ptr = reinterpret_cast<const half*>(gq + (tileM * D + k0));
        if (tid == 0)
            CP_ASYNC_BULK_TENSOR_2D(smem_q, gq_ptr, M_TILE, K_TILE * 2);
        __syncthreads();
        for (int m64 = 0; m64 < 2; ++m64)
            for (int n64 = 0; n64 < D / 64; ++n64) {
                uint64_t ptrA = __cvta_generic_to_shared(smem_gs + m64 * 64 * N_TILE); // grad_S^T
                uint64_t ptrB = __cvta_generic_to_shared(smem_q + n64 * 64 * K_TILE);
                WGMMA_M64N64K16_F32(accum, ptrA, ptrB, 1.0f);
            }
        __syncthreads();
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            int gRow = tileN + (tid >> 3) * 4 + i;
            int gCol = tileM + (tid & 7) * 4 + j;
            if (gRow < N && gCol < D)
                gdk[gRow * D + gCol] = accum[i];
        }
}

// -------------------- host 接口 --------------------
void flash_attn_v2_hopper_wgmma_backward(
    const float* q, const float* k, const float* v,
    const float* out, const float* dout,
    int B, int H, int N, int D)
{
    dim3 blk(WG, 1, 1);
    dim3 grd(1, (N + M_TILE - 1) / M_TILE, (N + N_TILE - 1) / N_TILE);
    size_t smem = (M_TILE * K_TILE + N_TILE * K_TILE + N_TILE * D +
                   M_TILE * N_TILE * 3) * 2;
    for (int bh = 0; bh < B * H; ++bh)
        flash_v2_hopper_bwd_kernel<<<grd, blk, smem>>>(
            q + bh * N * D, k + bh * N * D, v + bh * N * D,
            out + bh * N * D, dout + bh * N * D,
            nullptr, nullptr, nullptr, N, D);   // 实际工程需传 device 指针
    cudaDeviceSynchronize();
}


// // -------------------- 测试 main --------------------
// int main() {
//     int B = 1, H = 1, N = 256, D = 64;
//     size_t len = B * H * N * D;
//     size_t bytes = len * sizeof(float);

//     // host
//     float *hq_host = new float[len];
//     float *hk_host = new float[len];
//     float *hv_host = new float[len];
//     float *ho_host = new float[len];
//     for (size_t i = 0; i < len; ++i) {
//         float r = rand() / (float)RAND_MAX;
//         hq_host[i] = r;
//         hk_host[i] = r;
//         hv_host[i] = r;
//     }

//     // device
//     float *hq, *hk, *hv, *ho;
//     cudaMalloc(&hq, bytes);
//     cudaMalloc(&hk, bytes);
//     cudaMalloc(&hv, bytes);
//     cudaMalloc(&ho, bytes);
//     cudaMemcpy(hq, hq_host, bytes, cudaMemcpyHostToDevice);
//     cudaMemcpy(hk, hk_host, bytes, cudaMemcpyHostToDevice);
//     cudaMemcpy(hv, hv_host, bytes, cudaMemcpyHostToDevice);

//     // compute
//     flash_attn_v2_hopper_wgmma_fwd(hq, hk, hv, ho, B, H, N, D);

//     // copy back
//     cudaMemcpy(ho_host, ho, bytes, cudaMemcpyDeviceToHost);
//     printf("Output O (first 8x8):\n");
//     for (int i = 0; i < 8; ++i) {
//         for (int j = 0; j < 8; ++j)
//             printf("%8.5f ", ho_host[i * D + j]);
//         printf("\n");
//     }

//     // free
//     cudaFree(hq); cudaFree(hk); cudaFree(hv); cudaFree(ho);
//     delete[] hq_host; delete[] hk_host; delete[] hv_host; delete[] ho_host;
//     return 0;
// }