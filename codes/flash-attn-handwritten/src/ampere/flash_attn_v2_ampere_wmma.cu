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



// ----------- 设备端 kernel -----------
__global__ void flash_v2_ampere_wmma_backward_kernel(
    const float* q, const float* k, const float* v,
    const float* out, const float* dout,
    float* dQ, float* dK, float* dV,
    int N, int D)
{
    extern __shared__ float smem[];
    float* s   = smem;                       // [N,N]
    float* p   = s   + N*N;
    float* gs  = p   + N*N;
    half* hq   = reinterpret_cast<half*>(gs + N*N);
    half* hk   = hq  + WMMA_M * D;
    half* hv   = hk  + WMMA_N * D;

    int tid = threadIdx.x;
    int bid = blockIdx.x;   // 1 block per (B,H)
    int stride = N * D;
    q  += bid * stride; k  += bid * stride; v  += bid * stride;
    out+= bid * stride; dout+=bid * stride;
    dQ += bid * stride; dK += bid * stride; dV += bid * stride;

    // 1) S = QK^T  (WMMA)
    for (int tileM = 0; tileM < N; tileM += WMMA_M) {
        for (int tileN = 0; tileN < N; tileN += WMMA_N) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_q;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_k;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_s;
            wmma::fill_fragment(frag_s, 0.0f);

            // load Q,K → shared (简化：单线程拷贝)
            for (int i = tid; i < WMMA_M * D; i += blockDim.x)
                hq[i] = __float2half(q[tileM * D + i]);
            for (int i = tid; i < WMMA_N * D; i += blockDim.x)
                hk[i] = __float2half(k[tileN * D + i]);
            __syncthreads();

            for (int k0 = 0; k0 < D; k0 += WMMA_K) {
                wmma::load_matrix_sync(frag_q, hq + k0, D);
                wmma::load_matrix_sync(frag_k, hk + k0, D);
                wmma::mma_sync(frag_s, frag_q, frag_k, frag_s);
            }
            wmma::store_matrix_sync(&s[tileM * N + tileN], frag_s, N, wmma::mem_row_major);
        }
    }
    __syncthreads();

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
        if (tid == 0) shared_max = sum;   // 复用
        __syncthreads();
        for (int j = tid; j < N; j += blockDim.x)
            p[i * N + j] /= shared_max;
        __syncthreads();
    }

    // 3) dV = P^T @ dout
    for (int tileN = 0; tileN < N; tileN += WMMA_N) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_p;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_dout;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_dv;
        wmma::fill_fragment(frag_dv, 0.0f);

        // load P tile
        for (int i = tid; i < WMMA_M * N; i += blockDim.x)
            hq[i] = __float2half(p[i]);   // 复用 hq 做临时
        for (int i = tid; i < WMMA_M * D; i += blockDim.x)
            hk[i] = __float2half(dout[tileN * D + i]);
        __syncthreads();

        for (int k0 = 0; k0 < N; k0 += WMMA_K) {
            wmma::load_matrix_sync(frag_p, hq + k0 * N, N);
            wmma::load_matrix_sync(frag_dout, hk + k0, D);
            wmma::mma_sync(frag_dv, frag_p, frag_dout, frag_dv);
        }
        wmma::store_matrix_sync(&dV[tileN * D], frag_dv, D, wmma::mem_row_major);
    }
    __syncthreads();

    // 4) grad_S = dout @ V^T
    for (int tileM = 0; tileM < N; tileM += WMMA_M) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_dout;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_v;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_gs;
        wmma::fill_fragment(frag_gs, 0.0f);

        for (int i = tid; i < WMMA_M * D; i += blockDim.x)
            hq[i] = __float2half(dout[tileM * D + i]);
        for (int i = tid; i < WMMA_N * D; i += blockDim.x)
            hk[i] = __float2half(v[i]);
        __syncthreads();

        for (int k0 = 0; k0 < D; k0 += WMMA_K) {
            wmma::load_matrix_sync(frag_dout, hq + k0, D);
            wmma::load_matrix_sync(frag_v, hk + k0, D);
            wmma::mma_sync(frag_gs, frag_dout, frag_v, frag_gs);
        }
        wmma::store_matrix_sync(&gs[tileM * N], frag_gs, N, wmma::mem_row_major);
    }
    __syncthreads();

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
    for (int tileM = 0; tileM < N; tileM += WMMA_M) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_gs;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_dq;
        wmma::fill_fragment(frag_dq, 0.0f);

        for (int i = tid; i < WMMA_M * N; i += blockDim.x)
            hq[i] = __float2half(gs[tileM * N + i]);
        for (int i = tid; i < WMMA_N * D; i += blockDim.x)
            hk[i] = __float2half(k[i]);
        __syncthreads();

        for (int k0 = 0; k0 < D; k0 += WMMA_K) {
            wmma::load_matrix_sync(frag_gs, hq + k0 * N, N);
            wmma::load_matrix_sync(frag_k, hk + k0, D);
            wmma::mma_sync(frag_dq, frag_gs, frag_k, frag_dq);
        }
        wmma::store_matrix_sync(&dQ[tileM * D], frag_dq, D, wmma::mem_row_major);
    }
    __syncthreads();

    // 7) dK = grad_S^T @ Q
    for (int tileN = 0; tileN < N; tileN += WMMA_N) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_gs;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_q;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_dk;
        wmma::fill_fragment(frag_dk, 0.0f);

        for (int i = tid; i < WMMA_M * N; i += blockDim.x)
            hq[i] = __float2half(gs[i * N + tileN]);   // 转置
        for (int i = tid; i < WMMA_N * D; i += blockDim.x)
            hk[i] = __float2half(q[i]);
        __syncthreads();

        for (int k0 = 0; k0 < D; k0 += WMMA_K) {
            wmma::load_matrix_sync(frag_gs, hq + k0 * N, N);
            wmma::load_matrix_sync(frag_q, hk + k0, D);
            wmma::mma_sync(frag_dk, frag_gs, frag_q, frag_dk);
        }
        wmma::store_matrix_sync(&dK[tileN * D], frag_dk, D, wmma::mem_row_major);
    }
}

FlashAttnGrad flash_attn_v2_ampere_wmma_backward(const float* q, const float* k, const float* v,
                                          const float* out, const float* dout,
                                          int B, int H, int N, int D)
{
    size_t len = B * H * N * D;
    float *dQ, *dK, *dV;
    cudaMalloc(&dQ, len * sizeof(float));
    cudaMalloc(&dK, len * sizeof(float));
    cudaMalloc(&dV, len * sizeof(float));

    size_t smem = (3 * N * N + 3 * WMMA_M * D) * sizeof(float) + (WMMA_M * D + WMMA_N * D + WMMA_N * D) * sizeof(half);
    dim3 grid(B * H, 1, 1);
    dim3 block(256, 1, 1);
    flash_v2_ampere_wmma_backward_kernel<<<grid, block, smem>>>(
        q, k, v, out, dout, dQ, dK, dV, N, D);
    cudaDeviceSynchronize();
    return {dQ, dK, dV};
}