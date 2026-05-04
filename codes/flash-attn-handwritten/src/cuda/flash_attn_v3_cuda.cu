#include "cuda/flash_attn_v3_cuda.h"
#include "baseline/flash_attn_v2_baseline.h"
#include "common.h"
#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr int TILE_N = 64;
constexpr int BWD_TILE_N = 32;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
constexpr int SUBWARP_SIZE = 16;
constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * 2;
constexpr int MAX_D_PER_THREAD = 8;
constexpr int BWD_MAX_WARPS_PER_BLOCK = 16;

__device__ __forceinline__ float subwarp_reduce_sum(float value, int lane16)
{
    for (int offset = SUBWARP_SIZE / 2; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset, SUBWARP_SIZE);
    return value;
}

__device__ __forceinline__ float subwarp_reduce_max(float value)
{
    for (int offset = SUBWARP_SIZE / 2; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset, SUBWARP_SIZE));
    return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

} // namespace

__global__ void flash_v3_cuda_fwd_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ out,
    int N, int D)
{
    extern __shared__ float smem[];
    float* k_shared = smem;                                     // [TILE_N, D]
    float* v_shared = k_shared + TILE_N * D;                    // [TILE_N, D]
    float* probs = v_shared + TILE_N * D;                       // [ROWS_PER_BLOCK, TILE_N]
    float* row_max = probs + ROWS_PER_BLOCK * TILE_N;           // [ROWS_PER_BLOCK]
    float* row_sum = row_max + ROWS_PER_BLOCK;                  // [ROWS_PER_BLOCK]
    float* old_scale = row_sum + ROWS_PER_BLOCK;                // [ROWS_PER_BLOCK]

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int sub = lane >> 4;                                  // 0 or 1
    const int lane16 = lane & 15;
    const int row_slot = warp * 2 + sub;
    const int bh = blockIdx.x;
    const int row_base = blockIdx.y * ROWS_PER_BLOCK;
    const int row = row_base + row_slot;
    const bool active = row < N;
    const int stride = N * D;

    q += bh * stride;
    k += bh * stride;
    v += bh * stride;
    out += bh * stride;

    int d_indices[MAX_D_PER_THREAD];
    float q_values[MAX_D_PER_THREAD];
    float acc_values[MAX_D_PER_THREAD];
    int d_count = 0;

    if (active) {
        for (int d = lane16; d < D && d_count < MAX_D_PER_THREAD; d += SUBWARP_SIZE) {
            d_indices[d_count] = d;
            q_values[d_count] = q[row * D + d];
            acc_values[d_count] = 0.0f;
            ++d_count;
        }
    }
    if (lane16 == 0) {
        row_max[row_slot] = -INFINITY;
        row_sum[row_slot] = 0.0f;
        old_scale[row_slot] = 0.0f;
    }
    __syncthreads();

    for (int tile = 0; tile < N; tile += TILE_N) {
        const int actual = min(TILE_N, N - tile);

        if ((D & 3) == 0) {
            const int D4 = D / 4;
            for (int idx4 = tid; idx4 < actual * D4; idx4 += THREADS_PER_BLOCK) {
                const int jj = idx4 / D4;
                const int d4 = idx4 % D4;
                reinterpret_cast<float4*>(k_shared + jj * D)[d4] =
                    reinterpret_cast<const float4*>(k + (tile + jj) * D)[d4];
                reinterpret_cast<float4*>(v_shared + jj * D)[d4] =
                    reinterpret_cast<const float4*>(v + (tile + jj) * D)[d4];
            }
        } else {
            for (int idx = tid; idx < actual * D; idx += THREADS_PER_BLOCK) {
                const int jj = idx / D;
                const int d = idx % D;
                k_shared[idx] = k[(tile + jj) * D + d];
                v_shared[idx] = v[(tile + jj) * D + d];
            }
        }
        __syncthreads();

        if (active) {
            for (int jj = 0; jj < actual; ++jj) {
                float partial = 0.0f;
                for (int t = 0; t < d_count; ++t)
                    partial += q_values[t] * k_shared[jj * D + d_indices[t]];
                const float dot = subwarp_reduce_sum(partial, lane16);
                if (lane16 == 0)
                    probs[row_slot * TILE_N + jj] = dot;
            }
        }
        __syncthreads();

        if (active) {
            float tile_max = -INFINITY;
            for (int jj = lane16; jj < actual; jj += SUBWARP_SIZE)
                tile_max = fmaxf(tile_max, probs[row_slot * TILE_N + jj]);
            tile_max = subwarp_reduce_max(tile_max);
            tile_max = __shfl_sync(0xffffffffu, tile_max, 0, SUBWARP_SIZE);

            const float prev_max = row_max[row_slot];
            const float prev_sum = row_sum[row_slot];
            const float new_max = fmaxf(prev_max, tile_max);
            const float scale = prev_sum == 0.0f ? 0.0f : expf(prev_max - new_max);

            float local_sum = 0.0f;
            for (int jj = lane16; jj < actual; jj += SUBWARP_SIZE) {
                const float w = expf(probs[row_slot * TILE_N + jj] - new_max);
                probs[row_slot * TILE_N + jj] = w;
                local_sum += w;
            }
            float tile_sum = subwarp_reduce_sum(local_sum, lane16);
            tile_sum = __shfl_sync(0xffffffffu, tile_sum, 0, SUBWARP_SIZE);

            if (lane16 == 0) {
                row_max[row_slot] = new_max;
                row_sum[row_slot] = prev_sum * scale + tile_sum;
                old_scale[row_slot] = scale;
            }
        }
        __syncthreads();

        if (active) {
            const float scale = old_scale[row_slot];
            for (int t = 0; t < d_count; ++t) {
                float tile_acc = 0.0f;
                const int d = d_indices[t];
                for (int jj = 0; jj < actual; ++jj)
                    tile_acc += probs[row_slot * TILE_N + jj] * v_shared[jj * D + d];
                acc_values[t] = acc_values[t] * scale + tile_acc;
            }
        }
        __syncthreads();
    }

    if (active) {
        const float inv_sum = 1.0f / row_sum[row_slot];
        for (int t = 0; t < d_count; ++t)
            out[row * D + d_indices[t]] = acc_values[t] * inv_sum;
    }
}

void flash_attn_v3_cuda_fwd(const float* q, const float* k, const float* v,
                            float* out,
                            int B, int H, int N, int D)
{
    const dim3 grid(B * H, (N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1);
    const dim3 block(THREADS_PER_BLOCK, 1, 1);
    const size_t smem =
        (2 * TILE_N * D + ROWS_PER_BLOCK * TILE_N +
         3 * ROWS_PER_BLOCK) * sizeof(float);

    flash_v3_cuda_fwd_kernel<<<grid, block, smem>>>(q, k, v, out, N, D);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void flash_v3_cuda_backward_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ out, const float* __restrict__ dout,
    float* __restrict__ dQ, float* __restrict__ dK, float* __restrict__ dV,
    int N, int D, int rows_per_block)
{
    (void)out;

    extern __shared__ float smem[];
    float* k_shared = smem;                                             // [BWD_TILE_N, D]
    float* v_shared = k_shared + BWD_TILE_N * D;                        // [BWD_TILE_N, D]
    float* probs = v_shared + BWD_TILE_N * D;                           // [rows_per_block, BWD_TILE_N]
    float* gradp = probs + rows_per_block * BWD_TILE_N;                 // [rows_per_block, BWD_TILE_N]
    float* gs = gradp + rows_per_block * BWD_TILE_N;                    // [rows_per_block, BWD_TILE_N]
    float* row_max_buf = gs + rows_per_block * BWD_TILE_N;              // [rows_per_block]
    float* row_sum_buf = row_max_buf + rows_per_block;                  // [rows_per_block]
    float* row_softmax_sum_buf = row_sum_buf + rows_per_block;          // [rows_per_block]
    float* dV_tile = row_softmax_sum_buf + rows_per_block;              // [BWD_TILE_N, D]
    float* dK_tile = dV_tile + BWD_TILE_N * D;                          // [BWD_TILE_N, D]

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int bh = blockIdx.x;
    const int row_base = blockIdx.y * rows_per_block;
    const int row = row_base + warp;
    const bool active = warp < rows_per_block && row < N;

    const int stride = N * D;
    q += bh * stride;
    k += bh * stride;
    v += bh * stride;
    dout += bh * stride;
    dQ += bh * stride;
    dK += bh * stride;
    dV += bh * stride;

    const float* q_row = active ? q + row * D : nullptr;
    const float* dout_row = active ? dout + row * D : nullptr;
    float* dQ_row = active ? dQ + row * D : nullptr;

    int d_indices[MAX_D_PER_THREAD];
    float q_values[MAX_D_PER_THREAD];
    float dout_values[MAX_D_PER_THREAD];
    float dq_values[MAX_D_PER_THREAD];
    int d_count = 0;
    if (active) {
        for (int d = lane; d < D && d_count < MAX_D_PER_THREAD; d += 32) {
            d_indices[d_count] = d;
            q_values[d_count] = q_row[d];
            dout_values[d_count] = dout_row[d];
            dq_values[d_count] = 0.0f;
            ++d_count;
        }
    }

    float row_max = -INFINITY;
    float row_sum = 0.0f;

    for (int tile = 0; tile < N; tile += BWD_TILE_N) {
        const int actual = min(BWD_TILE_N, N - tile);

        if ((D & 3) == 0) {
            const int D4 = D / 4;
            for (int idx4 = lane; idx4 < actual * D4; idx4 += 32) {
                const int jj = idx4 / D4;
                const int d4 = idx4 % D4;
                reinterpret_cast<float4*>(k_shared + jj * D)[d4] =
                    reinterpret_cast<const float4*>(k + (tile + jj) * D)[d4];
            }
        } else {
            for (int idx = lane; idx < actual * D; idx += 32) {
                const int jj = idx / D;
                const int d = idx % D;
                k_shared[idx] = k[(tile + jj) * D + d];
            }
        }
        __syncthreads();

        for (int jj = 0; jj < actual; ++jj) {
            float partial = 0.0f;
            for (int t = 0; t < d_count; ++t)
                partial += q_values[t] * k_shared[jj * D + d_indices[t]];
            const float score = warp_reduce_sum(partial);
            if (lane == 0)
                probs[warp * BWD_TILE_N + jj] = score;
        }
        __syncthreads();

        float tile_max = -INFINITY;
        for (int jj = lane; jj < actual; jj += 32)
            tile_max = fmaxf(tile_max, probs[warp * BWD_TILE_N + jj]);
        for (int offset = 16; offset > 0; offset >>= 1)
            tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffffu, tile_max, offset));
        tile_max = __shfl_sync(0xffffffffu, tile_max, 0);

        const float new_max = fmaxf(row_max, tile_max);
        const float scale = row_sum == 0.0f ? 0.0f : expf(row_max - new_max);
        float local_sum = 0.0f;
        for (int jj = lane; jj < actual; jj += 32)
            local_sum += expf(probs[warp * BWD_TILE_N + jj] - new_max);
        const float tile_sum = warp_reduce_sum(local_sum);
        if (lane == 0) {
            row_max = new_max;
            row_sum = row_sum * scale + tile_sum;
            row_max_buf[warp] = row_max;
            row_sum_buf[warp] = row_sum;
        }
        __syncthreads();
        row_max = row_max_buf[warp];
        row_sum = row_sum_buf[warp];
    }

    float row_softmax_sum = 0.0f;
    for (int tile = 0; tile < N; tile += BWD_TILE_N) {
        const int actual = min(BWD_TILE_N, N - tile);

        if ((D & 3) == 0) {
            const int D4 = D / 4;
            for (int idx4 = tid; idx4 < actual * D4; idx4 += blockDim.x) {
                const int jj = idx4 / D4;
                const int d4 = idx4 % D4;
                reinterpret_cast<float4*>(k_shared + jj * D)[d4] =
                    reinterpret_cast<const float4*>(k + (tile + jj) * D)[d4];
                reinterpret_cast<float4*>(v_shared + jj * D)[d4] =
                    reinterpret_cast<const float4*>(v + (tile + jj) * D)[d4];
            }
        } else {
            for (int idx = tid; idx < actual * D; idx += blockDim.x) {
                const int jj = idx / D;
                const int d = idx % D;
                k_shared[idx] = k[(tile + jj) * D + d];
                v_shared[idx] = v[(tile + jj) * D + d];
            }
        }
        __syncthreads();

        for (int idx = tid; idx < actual * D; idx += blockDim.x)
            dV_tile[idx] = 0.0f;
        __syncthreads();

        for (int jj = 0; jj < actual; ++jj) {
            float partial = 0.0f;
            for (int t = 0; t < d_count; ++t)
                partial += q_values[t] * k_shared[jj * D + d_indices[t]];
            const float score = warp_reduce_sum(partial);
            if (lane == 0)
                probs[warp * BWD_TILE_N + jj] = expf(score - row_max) / row_sum;
        }
        __syncthreads();

        for (int jj = 0; jj < actual; ++jj) {
            float partial = 0.0f;
            for (int t = 0; t < d_count; ++t)
                partial += dout_values[t] * v_shared[jj * D + d_indices[t]];
            const float g = warp_reduce_sum(partial);
            if (lane == 0)
                gradp[warp * BWD_TILE_N + jj] = g;
        }
        __syncthreads();

        for (int t = 0; t < d_count; ++t) {
            const int d = d_indices[t];
            for (int jj = 0; jj < actual; ++jj)
                atomicAdd(&dV_tile[jj * D + d], probs[warp * BWD_TILE_N + jj] * dout_values[t]);
        }
        __syncthreads();

        float local_softmax_sum = 0.0f;
        for (int jj = lane; jj < actual; jj += 32)
            local_softmax_sum += gradp[warp * BWD_TILE_N + jj] * probs[warp * BWD_TILE_N + jj];
        const float tile_softmax_sum = warp_reduce_sum(local_softmax_sum);
        if (lane == 0) {
            row_softmax_sum += tile_softmax_sum;
            row_softmax_sum_buf[warp] = row_softmax_sum;
        }
        __syncthreads();

        for (int idx = tid; idx < actual * D; idx += blockDim.x)
            atomicAdd(&dV[(tile * D) + idx], dV_tile[idx]);
        __syncthreads();
    }

    row_softmax_sum = row_softmax_sum_buf[warp];

    for (int tile = 0; tile < N; tile += BWD_TILE_N) {
        const int actual = min(BWD_TILE_N, N - tile);

        if ((D & 3) == 0) {
            const int D4 = D / 4;
            for (int idx4 = tid; idx4 < actual * D4; idx4 += blockDim.x) {
                const int jj = idx4 / D4;
                const int d4 = idx4 % D4;
                reinterpret_cast<float4*>(k_shared + jj * D)[d4] =
                    reinterpret_cast<const float4*>(k + (tile + jj) * D)[d4];
                reinterpret_cast<float4*>(v_shared + jj * D)[d4] =
                    reinterpret_cast<const float4*>(v + (tile + jj) * D)[d4];
            }
        } else {
            for (int idx = tid; idx < actual * D; idx += blockDim.x) {
                const int jj = idx / D;
                const int d = idx % D;
                k_shared[idx] = k[(tile + jj) * D + d];
                v_shared[idx] = v[(tile + jj) * D + d];
            }
        }
        __syncthreads();

        for (int idx = tid; idx < actual * D; idx += blockDim.x)
            dK_tile[idx] = 0.0f;
        __syncthreads();

        for (int jj = 0; jj < actual; ++jj) {
            float partial = 0.0f;
            for (int t = 0; t < d_count; ++t)
                partial += q_values[t] * k_shared[jj * D + d_indices[t]];
            const float score = warp_reduce_sum(partial);
            if (lane == 0)
                probs[warp * BWD_TILE_N + jj] = expf(score - row_max) / row_sum;
        }
        __syncthreads();

        for (int jj = 0; jj < actual; ++jj) {
            float partial = 0.0f;
            for (int t = 0; t < d_count; ++t)
                partial += dout_values[t] * v_shared[jj * D + d_indices[t]];
            const float g = warp_reduce_sum(partial);
            if (lane == 0)
                gradp[warp * BWD_TILE_N + jj] = g;
        }
        __syncthreads();

        for (int jj = lane; jj < actual; jj += 32)
            gs[warp * BWD_TILE_N + jj] =
                probs[warp * BWD_TILE_N + jj] * (gradp[warp * BWD_TILE_N + jj] - row_softmax_sum);
        __syncthreads();

        for (int t = 0; t < d_count; ++t) {
            const int d = d_indices[t];
            for (int jj = 0; jj < actual; ++jj) {
                atomicAdd(&dK_tile[jj * D + d], gs[warp * BWD_TILE_N + jj] * q_values[t]);
                dq_values[t] += gs[warp * BWD_TILE_N + jj] * k_shared[jj * D + d];
            }
        }
        __syncthreads();

        for (int idx = tid; idx < actual * D; idx += blockDim.x)
            atomicAdd(&dK[(tile * D) + idx], dK_tile[idx]);
        __syncthreads();
    }

    for (int t = 0; t < d_count; ++t)
        dQ_row[d_indices[t]] = dq_values[t];
}

FlashAttnGrad flash_attn_v3_cuda_backward(const float* q, const float* k, const float* v,
                                          const float* out, const float* dout,
                                          int B, int H, int N, int D)
{
    size_t len = static_cast<size_t>(B) * H * N * D;
    float *dQ, *dK, *dV;
    CHECK_CUDA(cudaMalloc(&dQ, len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dK, len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dV, len * sizeof(float)));
    CHECK_CUDA(cudaMemset(dQ, 0, len * sizeof(float)));
    CHECK_CUDA(cudaMemset(dK, 0, len * sizeof(float)));
    CHECK_CUDA(cudaMemset(dV, 0, len * sizeof(float)));

    const int rows_per_block = N >= 512 ? 16 : 8;
    const dim3 grid(B * H, (N + rows_per_block - 1) / rows_per_block, 1);
    const dim3 block(rows_per_block * 32, 1, 1);
    const size_t smem =
        (2 * BWD_TILE_N * D + 3 * rows_per_block * BWD_TILE_N +
         3 * rows_per_block + 2 * BWD_TILE_N * D) * sizeof(float);

    int max_optin_smem = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if (static_cast<int>(smem) > max_optin_smem) {
        fprintf(stderr, "flash_attn_v3_cuda_backward: required shared memory %zu exceeds opt-in limit %d\n",
                smem, max_optin_smem);
        exit(EXIT_FAILURE);
    }
    if (static_cast<int>(smem) > 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(
            flash_v3_cuda_backward_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem)));
    }

    flash_v3_cuda_backward_kernel<<<grid, block, smem>>>(q, k, v, out, dout, dQ, dK, dV, N, D, rows_per_block);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    return {dQ, dK, dV};
}
