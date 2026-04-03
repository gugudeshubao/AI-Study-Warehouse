#include "cpu/flash_attn_v3_cpu.h"
#include "cpu/flash_attn_v2_cpu.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace {

constexpr int TILE_N = 32;

} // namespace

void flash_attn_v3_cpu_fwd(const float* q, const float* k, const float* v,
                           float* out,
                           int B, int H, int N, int D)
{
    const int stride = N * D;
    std::vector<float> scores(TILE_N);
    std::vector<float> weights(TILE_N);
    std::vector<float> acc(D);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            const float* q_ptr = q + (b * H + h) * stride;
            const float* k_ptr = k + (b * H + h) * stride;
            const float* v_ptr = v + (b * H + h) * stride;
            float* o_ptr = out + (b * H + h) * stride;

            for (int i = 0; i < N; ++i) {
                float running_max = -INFINITY;
                float running_sum = 0.0f;
                std::fill(acc.begin(), acc.end(), 0.0f);

                for (int tile = 0; tile < N; tile += TILE_N) {
                    const int actual = std::min(TILE_N, N - tile);
                    float tile_max = -INFINITY;

                    for (int jj = 0; jj < actual; ++jj) {
                        const int j = tile + jj;
                        float dot = 0.0f;
                        for (int d = 0; d < D; ++d)
                            dot += q_ptr[i * D + d] * k_ptr[j * D + d];
                        scores[jj] = dot;
                        tile_max = std::max(tile_max, dot);
                    }

                    const float new_max = std::max(running_max, tile_max);
                    const float old_scale = running_sum == 0.0f ? 0.0f : std::exp(running_max - new_max);
                    float new_sum = running_sum * old_scale;

                    for (int jj = 0; jj < actual; ++jj) {
                        weights[jj] = std::exp(scores[jj] - new_max);
                        new_sum += weights[jj];
                    }

                    for (int d = 0; d < D; ++d) {
                        float tile_acc = 0.0f;
                        for (int jj = 0; jj < actual; ++jj)
                            tile_acc += weights[jj] * v_ptr[(tile + jj) * D + d];
                        acc[d] = acc[d] * old_scale + tile_acc;
                    }

                    running_max = new_max;
                    running_sum = new_sum;
                }

                for (int d = 0; d < D; ++d)
                    o_ptr[i * D + d] = acc[d] / running_sum;
            }
        }
    }
}

FlashAttnGrad flash_attn_v3_cpu_backward(const float* q, const float* k, const float* v,
                                         const float* out, const float* dout,
                                         int B, int H, int N, int D)
{
    return flash_attn_v2_cpu_backward(q, k, v, out, dout, B, H, N, D);
}
