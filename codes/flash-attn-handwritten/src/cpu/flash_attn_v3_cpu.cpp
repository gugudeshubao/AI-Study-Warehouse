#include "cpu/flash_attn_v3_cpu.h"
#include <cmath>
#include <vector>
#include <cuda_runtime.h>  // 仅为了使用 cudaMallocHost / cudaFree

constexpr int TILE_N = 32;   // 可按需调大

void flash_attn_v3_cpu_fwd(const float* q, const float* k, const float* v,
                           float* out,
                           int B, int H, int N, int D)
{
    size_t len = B * H * N * D;
    int stride = N * D;
    std::vector<float> tile_p(TILE_N), tile_sum(TILE_N);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            const float* q_ptr = q + (b * H + h) * stride;
            const float* k_ptr = k + (b * H + h) * stride;
            const float* v_ptr = v + (b * H + h) * stride;
            float* o_ptr = out + (b * H + h) * stride;

            // 逐行 online softmax
            for (int i = 0; i < N; ++i) {
                float row_max = -std::numeric_limits<float>::infinity();
                float row_sum = 0.0f;
                std::fill(tile_sum.begin(), tile_sum.end(), 0.0f);

                // 分 tile 处理
                for (int tile = 0; tile < N; tile += TILE_N) {
                    int actual = std::min(TILE_N, N - tile);
                    // 1) 计算 tile 内 S[i][j]
                    for (int jj = 0; jj < actual; ++jj) {
                        int j = tile + jj;
                        float dot = 0.0f;
                        for (int d = 0; d < D; ++d)
                            dot += q_ptr[i * D + d] * k_ptr[j * D + d];
                        tile_p[jj] = dot;
                    }
                    // 2) online max
                    for (int jj = 0; jj < actual; ++jj)
                        row_max = std::max(row_max, tile_p[jj]);
                    // 3) online sum
                    for (int jj = 0; jj < actual; ++jj)
                        row_sum += std::exp(tile_p[jj] - row_max);
                    // 4) running O += exp(S-row_max) * V
                    for (int d = 0; d < D; ++d) {
                        float pv = 0.0f;
                        for (int jj = 0; jj < actual; ++jj)
                            pv += std::exp(tile_p[jj] - row_max) * v_ptr[(tile + jj) * D + d];
                        o_ptr[i * D + d] += pv;
                    }
                }
                // 5) 归一化
                for (int d = 0; d < D; ++d)
                    o_ptr[i * D + d] /= row_sum;
            }
        }
    }
}

#include <cuda_runtime.h>

constexpr int TILE_N = 32;

void softmax_backward_row(float* grad, const float* out, int n) {
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) sum += grad[j] * out[j];
    for (int j = 0; j < n; ++j) grad[j] = out[j] * (grad[j] - sum);
}

FlashAttnGrad flash_attn_v3_cpu_backward(const float* q, const float* k, const float* v,
                                         const float* out, const float* dout,
                                         int B, int H, int N, int D)
{
    size_t len = B * H * N * D;
    float *dQ, *dK, *dV;
    cudaMallocHost(&dQ, len * sizeof(float));
    cudaMallocHost(&dK, len * sizeof(float));
    cudaMallocHost(&dV, len * sizeof(float));
    std::fill(dQ, dQ + len, 0.0f);
    std::fill(dK, dK + len, 0.0f);
    std::fill(dV, dV + len, 0.0f);

    int stride = N * D;
    std::vector<float> tile_p(TILE_N), tile_sum(TILE_N), tile_max(TILE_N);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            const float* q_ptr = q + (b * H + h) * stride;
            const float* k_ptr = k + (b * H + h) * stride;
            const float* v_ptr = v + (b * H + h) * stride;
            const float* o_ptr = out + (b * H + h) * stride;
            const float* dout_ptr = dout + (b * H + h) * stride;
            float* dq_ptr = dQ + (b * H + h) * stride;
            float* dk_ptr = dK + (b * H + h) * stride;
            float* dv_ptr = dV + (b * H + h) * stride;

            // 1) 前向 S = QK^T (复用，仅维护 row_max, row_sum)
            std::vector<float> row_max(N, -std::numeric_limits<float>::infinity());
            std::vector<float> row_sum(N, 0.0f);
            for (int i = 0; i < N; ++i) {
                for (int tile = 0; tile < N; tile += TILE_N) {
                    int actual = std::min(TILE_N, N - tile);
                    // tile max
                    for (int jj = 0; jj < actual; ++jj) {
                        int j = tile + jj;
                        float dot = 0.0f;
                        for (int d = 0; d < D; ++d)
                            dot += q_ptr[i * D + d] * k_ptr[j * D + d];
                        tile_p[jj] = dot;
                        row_max[i] = std::max(row_max[i], dot);
                    }
                    // tile sum
                    for (int jj = 0; jj < actual; ++jj)
                        row_sum[i] += std::exp(tile_p[jj] - row_max[i]);
                }
            }

            // 2) dV = P^T @ dout  (online)
            for (int i = 0; i < N; ++i) {
                for (int tile = 0; tile < N; tile += TILE_N) {
                    int actual = std::min(TILE_N, N - tile);
                    // tile_p = exp(S - row_max) / row_sum
                    for (int jj = 0; jj < actual; ++jj) {
                        int j = tile + jj;
                        float dot = 0.0f;
                        for (int d = 0; d < D; ++d)
                            dot += q_ptr[i * D + d] * k_ptr[j * D + d];
                        tile_p[jj] = std::exp(dot - row_max[i]) / row_sum[i];
                    }
                    // running dV += P^T * dout
                    for (int d = 0; d < D; ++d) {
                        float pv = 0.0f;
                        for (int jj = 0; jj < actual; ++jj)
                            pv += tile_p[jj] * dout_ptr[i * D + d];
                        dv_ptr[j * D + d] += pv;
                    }
                }
            }

            // 3) grad_S = dout @ V^T  (online)
            std::vector<float> grad_s_row(N);
            for (int i = 0; i < N; ++i) {
                std::fill(grad_s_row.begin(), grad_s_row.end(), 0.0f);
                for (int tile = 0; tile < N; tile += TILE_N) {
                    int actual = std::min(TILE_N, N - tile);
                    // tile_p = exp(S - row_max) / row_sum
                    for (int jj = 0; jj < actual; ++jj) {
                        int j = tile + jj;
                        float dot = 0.0f;
                        for (int d = 0; d < D; ++d)
                            dot += q_ptr[i * D + d] * k_ptr[j * D + d];
                        tile_p[jj] = std::exp(dot - row_max[i]) / row_sum[i];
                    }
                    // running grad_S += dout * V^T
                    for (int jj = 0; jj < actual; ++jj) {
                        float pv = 0.0f;
                        for (int d = 0; d < D; ++d)
                            pv += dout_ptr[i * D + d] * v_ptr[(tile + jj) * D + d];
                        grad_s_row[tile + jj] += pv;
                    }
                }
                // softmax 反向
                softmax_backward_row(&grad_s_row[0], &tile_p[0], N);
                // 4) dQ = grad_S @ K
                for (int d = 0; d < D; ++d) {
                    float sum = 0.0f;
                    for (int j = 0; j < N; ++j)
                        sum += grad_s_row[j] * k_ptr[j * D + d];
                    dq_ptr[i * D + d] += sum;
                }
                // 5) dK = grad_S^T @ Q
                for (int tile = 0; tile < N; tile += TILE_N) {
                    int actual = std::min(TILE_N, N - tile);
                    for (int jj = 0; jj < actual; ++jj) {
                        int j = tile + jj;
                        float sum = 0.0f;
                        for (int d = 0; d < D; ++d)
                            sum += grad_s_row[i] * q_ptr[i * D + d];
                        dk_ptr[j * D + d] += sum;
                    }
                }
            }
        }
    }
    return {dQ, dK, dV};
}
