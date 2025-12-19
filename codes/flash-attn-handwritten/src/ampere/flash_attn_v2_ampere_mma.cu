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




// ----------- 设备端 kernel -----------
__global__ void flash_v2_ampere_mma_backward_kernel(
    const float* q, const float* k, const float* v,
    const float* out, const float* dout,
    float* dQ, float* dK, float* dV,
    int N, int D)
{
    extern __shared__ char smem[];
    float* s  = reinterpret_cast<float*>(smem);
    float* p  = s  + N * N;
    float* gs = p  + N * N;
    half* hq  = reinterpret_cast<half*>(gs + N * N);
    half* hk  = hq + MMA_M * D;
    half* hv  = hk + MMA_N * D;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int stride = N * D;
    q  += bid * stride; k  += bid * stride; v  += bid * stride;
    out+= bid * stride; dout+=bid * stride;
    dQ += bid * stride; dK += bid * stride; dV += bid * stride;

    // 1) S = QK^T  (mma)
    for (int tileM = 0; tileM < N; tileM += MMA_M) {
        for (int tileN = 0; tileN < N; tileN += MMA_N) {
            // load Q,K → shared (cp.async)
            for (int i = tid; i < MMA_M * D; i += blockDim.x) {
                int row = i / D, col = i % D;
                __pipeline_memcpy_async(&hq[i], &q[tileM * D + i], sizeof(half));
            }
            for (int i = tid; i < MMA_N * D; i += blockDim.x) {
                int row = i / D, col = i % D;
                __pipeline_memcpy_async(&hk[i], &k[tileN * D + i], sizeof(half));
            }
            __pipeline_commit();
            __pipeline_wait_prior(0);
            __syncthreads();

            // mma
            uint32_t a[2], b[1];
            float c[4] = {};
            for (int k0 = 0; k0 < D; k0 += MMA_K) {
                // ldmatrix
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                             :: "r"(a[0]), "r"(a[1]), "l"(hq + k0));
                asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                             :: "r"(b[0]), "l"(hk + k0));
                // mma
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                             "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                             : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                             : "r"(a[0]), "r"(a[1]), "r"(b[0]));
            }
            // store
            for (int i = 0; i < 4; ++i)
                s[(tileM + (tid / 32) * 4 + i) * N + tileN + (tid & 3) * 4 + i] = c[i];
            __syncthreads();
        }
    }

    // 2) P = softmax(S)
    for (int i = 0; i < N; ++i) {
        float maxVal = -1e38f;
        for (int j = tid; j < N; j += blockDim.x)
            maxVal = fmaxf(maxVal, s[i * N + j]);
        __shared__ float shared_max;
        __syncthreads();
        if (tid == 0) shared_max = maxVal;
        __syncthreads();
        float sum = 0.0f;
        for (int j = tid; j < N; j += blockDim.x) {
            float ev = expf(s[i * N + j] - shared_max);
            p[i * N + j] = ev;
            sum += ev;
        }
        __syncthreads();
        if (tid == 0) shared_max = sum;
        __syncthreads();
        for (int j = tid; j < N; j += blockDim.x)
            p[i * N + j] /= shared_max;
        __syncthreads();
    }

    // 3) dV = P^T @ dout  (mma)
    for (int tileN = 0; tileN < N; tileN += MMA_N) {
        // load P, dout → shared
        for (int i = tid; i < MMA_M * N; i += blockDim.x)
            __pipeline_memcpy_async(&hq[i], &p[i], sizeof(half));
        for (int i = tid; i < MMA_M * D; i += blockDim.x)
            __pipeline_memcpy_async(&hk[i], &dout[tileN * D + i], sizeof(half));
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        float c[4] = {};
        for (int k0 = 0; k0 < N; k0 += MMA_K) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                         :: "r"(a[0]), "r"(a[1]), "l"(hq + k0 * N));
            asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                         :: "r"(b[0]), "l"(hk + k0));
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                         : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                         : "r"(a[0]), "r"(a[1]), "r"(b[0]));
        }
        for (int i = 0; i < 4; ++i)
            dV[tileN + (tid / 32) * 4 + i + (tid & 3) * 4] = c[i];
        __syncthreads();
    }

    // 4) grad_S = dout @ V^T
    for (int tileM = 0; tileM < N; tileM += MMA_M) {
        for (int i = tid; i < MMA_M * D; i += blockDim.x)
            __pipeline_memcpy_async(&hq[i], &dout[tileM * D + i], sizeof(half));
        for (int i = tid; i < MMA_N * D; i += blockDim.x)
            __pipeline_memcpy_async(&hk[i], &v[i], sizeof(half));
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        float c[4] = {};
        for (int k0 = 0; k0 < D; k0 += MMA_K) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                         :: "r"(a[0]), "r"(a[1]), "l"(hq + k0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                         :: "r"(b[0]), "l"(hk + k0));
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                         : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                         : "r"(a[0]), "r"(a[1]), "r"(b[0]));
        }
        for (int i = 0; i < 4; ++i)
            gs[(tileM + (tid / 32) * 4 + i) * N + (tid & 3) * 4 + i] = c[i];
        __syncthreads();
    }

    // 5) softmax 反向
    for (int i = 0; i < N; ++i) {
        if (tid < N) {
            float sum = 0.0f;
            for (int j = 0; j < N; ++j) sum += gs[i * N + j] * p[i * N + j];
            for (int j = tid; j < N; j += blockDim.x)
                gs[i * N + j] = p[i * N + j] * (gs[i * N + j] - sum);
        }
        __syncthreads();
    }

    // 6) dQ = grad_S @ K
    for (int tileM = 0; tileM < N; tileM += MMA_M) {
        for (int i = tid; i < MMA_M * N; i += blockDim.x)
            __pipeline_memcpy_async(&hq[i], &gs[tileM * N + i], sizeof(half));
        for (int i = tid; i < MMA_N * D; i += blockDim.x)
            __pipeline_memcpy_async(&hk[i], &k[i], sizeof(half));
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        float c[4] = {};
        for (int k0 = 0; k0 < D; k0 += MMA_K) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                         :: "r"(a[0]), "r"(a[1]), "l"(hq + k0 * N));
            asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                         :: "r"(b[0]), "l"(hk + k0));
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                         : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                         : "r"(a[0]), "r"(a[1]), "r"(b[0]));
        }
        for (int i = 0; i < 4; ++i)
            dQ[tileM + (tid / 32) * 4 + i + (tid & 3) * 4] = c[i];
        __syncthreads();
    }

    // 7) dK = grad_S^T @ Q
    for (int tileN = 0; tileN < N; tileN += MMA_N) {
        for (int i = tid; i < MMA_M * N; i += blockDim.x)
            __pipeline_memcpy_async(&hq[i], &gs[i * N + tileN], sizeof(half));
        for (int i = tid; i < MMA_N * D; i += blockDim.x)
            __pipeline_memcpy_async(&hk[i], &q[i], sizeof(half));
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        float c[4] = {};
        for (int k0 = 0; k0 < D; k0 += MMA_K) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                         :: "r"(a[0]), "r"(a[1]), "l"(hq + k0 * N));
            asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                         :: "r"(b[0]), "l"(hk + k0));
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                         : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                         : "r"(a[0]), "r"(a[1]), "r"(b[0]));
        }
        for (int i = 0; i < 4; ++i)
            dK[tileN + (tid / 32) * 4 + i + (tid & 3) * 4] = c[i];
        __syncthreads();
    }
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

FlashAttnGrad flash_attn_v2_ampere_mma_backward(const float* q, const float* k, const float* v,
                                         const float* out, const float* dout,
                                         int B, int H, int N, int D)
{
        using namespace ampere_mma;
    size_t len = B * H * N * D;
    float *dQ, *dK, *dV;
    cudaMalloc(&dQ, len * sizeof(float));
    cudaMalloc(&dK, len * sizeof(float));
    cudaMalloc(&dV, len * sizeof(float));

    size_t smem = (3 * N * N + 3 * MMA_M * D) * sizeof(float) + (MMA_M * D + MMA_N * D + MMA_N * D) * sizeof(half);
    dim3 grid(B * H, 1, 1);
    dim3 block(256, 1, 1);
    flash_v2_ampere_mma_backward_kernel<<<grid, block, smem>>>(
        q, k, v, out, dout, dQ, dK, dV, N, D);
    cudaDeviceSynchronize();
    return {dQ, dK, dV};
}