#include "ampere/flash_attn_v3_ampere_mma.h"
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda/barrier>


constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;
constexpr int BLOCK = 128;

__global__ void flash_v3_ampere_mma_fwd_kernel(
    const float* q, const float* k, const float* v,
    float* out,
    int N, int D)
{
    extern __shared__ char smem[];
    half* hq = reinterpret_cast<half*>(smem);
    half* hk = hq + MMA_M * D;
    half* hv = hk + MMA_N * D;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int stride = N * D;
    q  += bid * stride; k  += bid * stride; v  += bid * stride;
    out+= bid * stride;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N) return;

    float row_max = -1e38f;
    float row_sum = 0.0f;

    for (int tile = 0; tile < N; tile += MMA_N) {
        int actual = min(MMA_N, N - tile);
        // 1) cp.async 加载 Q tile
        for (int i = tid; i < MMA_M * D; i += blockDim.x) {
            int r = i / D, c = i % D;
            __pipeline_memcpy_async(&hq[i], &q[row * D + i], sizeof(half));
        }
        // 2) cp.async 加载 K tile
        for (int i = tid; i < MMA_N * D; i += blockDim.x) {
            int r = i / D, c = i % D;
            __pipeline_memcpy_async(&hk[i], &k[tile * D + i], sizeof(half));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        // 3) S = Q@K^T  (mma.m16n8k16)
        uint32_t a[2], b[1];
        float c[4] = {};
        for (int k0 = 0; k0 < D; k0 += MMA_K) {
            // ldmatrix
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                         :: "r"(a[0]), "r"(a[1]), "l"(hq + k0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                         :: "r"(b[0]), "l"(hk + k0));
            // mma
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                         : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                         : "r"(a[0]), "r"(a[1]), "r"(b[0]));
        }
        __syncthreads();

        // 4) cp.async 加载 V tile
        for (int i = tid; i < MMA_N * D; i += blockDim.x) {
            int r = i / D, c = i % D;
            __pipeline_memcpy_async(&hv[i], &v[tile * D + i], sizeof(half));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        // 5) online softmax + running O
        for (int j = 0; j < actual; ++j) {
            float s_val = c[j];
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

void flash_attn_v3_ampere_mma_fwd(const float* q, const float* k, const float* v,
                           float* out,
                           int B, int H, int N, int D)
{
    dim3 block(128);
    dim3 grid(B * H, (N + 3) / 4, 1);
    size_t smem = (MMA_M * D + MMA_N * D + MMA_N * D) * sizeof(half);
    flash_v3_ampere_mma_fwd_kernel<<<grid, block, smem>>>(q, k, v, out, N, D);
    cudaDeviceSynchronize();
}



// -------------- 设备端 kernel --------------
__global__ void flash_v3_ampere_mma_backward_kernel(
    const float* q, const float* k, const float* v,
    const float* out, const float* dout,
    float* dQ, float* dK, float* dV,
    int N, int D)
{
    extern __shared__ char smem[];
    half* hq  = reinterpret_cast<half*>(smem);
    half* hk  = hq  + MMA_M * D;
    half* hv  = hk  + MMA_N * D;
    // 后面留 128 float 给 softmax reduce
    float* red = reinterpret_cast<float*>(hv + MMA_N * D);

    int tid  = threadIdx.x;
    int bid  = blockIdx.x;            // 1 block per (B,H)
    int stride = N * D;
    q   += bid * stride; k   += bid * stride; v   += bid * stride;
    out += bid * stride; dout+=bid * stride;
    dQ  += bid * stride; dK  += bid * stride; dV  += bid * stride;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N) return;

    float row_max = -1e38f;
    float row_sum = 0.0f;

    // 1) 前向 S = QK^T （仅维护 row_max, row_sum）
    for (int tile = 0; tile < N; tile += MMA_N) {
        int actual = min(MMA_N, N - tile);
        // cp.async 加载 Q tile
        for (int i = tid; i < MMA_M * D; i += blockDim.x)
            __pipeline_memcpy_async(&hq[i], &q[row * D + i], sizeof(half));
        // cp.async 加载 K tile
        for (int i = tid; i < MMA_N * D; i += blockDim.x)
            __pipeline_memcpy_async(&hk[i], &k[tile * D + i], sizeof(half));
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        // mma.m16n8k16
        uint32_t a[2], b[1];
        float c[4] = {};
        for (int k0 = 0; k0 < D; k0 += MMA_K) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                         :: "r"(a[0]), "r"(a[1]), "l"(hq + k0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                         :: "r"(b[0]), "l"(hk + k0));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                         : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                         : "r"(a[0]), "r"(a[1]), "r"(b[0]));
        }
        __syncthreads();

        // online max & sum
        for (int j = 0; j < actual; ++j) {
            float s_val = c[j];
            row_max = fmaxf(row_max, s_val);
            float exp_s = expf(s_val - row_max);
            row_sum += exp_s;
        }
    }

    // 2) dV = P^T @ dout  (online)
    for (int tile = 0; tile < N; tile += MMA_N) {
        int actual = min(MMA_N, N - tile);
        // cp.async 加载 K tile
        for (int i = tid; i < MMA_N * D; i += blockDim.x)
            __pipeline_memcpy_async(&hk[i], &k[tile * D + i], sizeof(half));
        // cp.async 加载 V tile
        for (int i = tid; i < MMA_N * D; i += blockDim.x)
            __pipeline_memcpy_async(&hv[i], &v[tile * D + i], sizeof(half));
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        // S = QK^T  (复用 mma)
        uint32_t a[2], b[1];
        float c[4] = {};
        for (int k0 = 0; k0 < D; k0 += MMA_K) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                         :: "r"(a[0]), "r"(a[1]), "l"(hq + k0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                         :: "r"(b[0]), "l"(hk + k0));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                         : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                         : "r"(a[0]), "r"(a[1]), "r"(b[0]));
        }
        __syncthreads();

        // online P = softmax(S)
        for (int j = 0; j < actual; ++j) {
            float s_val = c[j];
            float exp_s = expf(s_val - row_max) / row_sum;
            for (int d = tid; d < D; d += blockDim.x) {
                float v_val = __half2float(hv[j * D + d]);
                float dout_val = dout[row * D + d];
                atomicAdd(&dV[(tile + j) * D + d], exp_s * dout_val);
            }
        }
    }

    // 3) grad_S = dout @ V^T  (online)
    for (int tile = 0; tile < N; tile += MMA_N) {
        int actual = min(MMA_N, N - tile);
        // cp.async 加载 V tile
        for (int i = tid; i < MMA_N * D; i += blockDim.x)
            __pipeline_memcpy_async(&hv[i], &v[tile * D + i], sizeof(half));
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        // S = QK^T  (复用 mma)
        uint32_t a[2], b[1];
        float c[4] = {};
        for (int k0 = 0; k0 < D; k0 += MMA_K) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                         :: "r"(a[0]), "r"(a[1]), "l"(hq + k0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                         :: "r"(b[0]), "l"(hk + k0));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                         : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                         : "r"(a[0]), "r"(a[1]), "r"(b[0]));
        }
        __syncthreads();

        // online grad_S = dout * V^T
        for (int j = 0; j < actual; ++j) {
            float s_val = c[j];
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
    for (int tile = 0; tile < N; tile += MMA_N) {
        int actual = min(MMA_N, N - tile);
        // cp.async 加载 Q tile
        for (int i = tid; i < MMA_M * D; i += blockDim.x)
            __pipeline_memcpy_async(&hq[i], &q[row * D + i], sizeof(half));
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        // S = QK^T  (复用 mma)
        uint32_t a[2], b[1];
        float c[4] = {};
        for (int k0 = 0; k0 < D; k0 += MMA_K) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                         :: "r"(a[0]), "r"(a[1]), "l"(hq + k0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                         :: "r"(b[0]), "l"(hk + k0));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                         : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                         : "r"(a[0]), "r"(a[1]), "r"(b[0]));
        }
        __syncthreads();

        // online grad_S^T @ Q
        for (int j = 0; j < actual; ++j) {
            float s_val = c[j];
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

// -------------- host wrapper --------------
FlashAttnGrad flash_attn_v3_ampere_mma_backward(const float* q, const float* k, const float* v,
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

    size_t smem = (MMA_M * D + MMA_N * D + MMA_N * D) * sizeof(half) + 128 * sizeof(float);
    dim3 grid(B * H, (N + 3) / 4, 1);
    dim3 block(128, 4, 1);
    flash_v3_ampere_mma_backward_kernel<<<grid, block, smem>>>(
        q, k, v, out, dout, dQ, dK, dV, N, D);
    cudaDeviceSynchronize();
    return {dQ, dK, dV};
}