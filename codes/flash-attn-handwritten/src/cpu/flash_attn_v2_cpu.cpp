#include "cpu/flash_attn_v2_cpu.h"
#include <cmath>
#include <vector>

void flash_attn_v2_cpu_fwd(const float* q, const float* k, const float* v,
                        float* out,
                        int B, int H, int N, int D)
{
    int stride = N * D;
    for(int b = 0; b < B; ++b)
    for(int h = 0; h < H; ++h)
    {
        const float* q_ptr = q + b * H * stride + h * stride;   // [N,D]
        const float* k_ptr = k + b * H * stride + h * stride;
        const float* v_ptr = v + b * H * stride + h * stride;
        float*       o_ptr = out + b * H * stride + h * stride;

        // S = QK^T
        std::vector<float> s(N * N, 0.0f);
        for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for(int d = 0; d < D; ++d)
                sum += q_ptr[i * D + d] * k_ptr[j * D + d];
            s[i * N + j] = sum;
        }
        // row-wise max for numerical stable softmax
        std::vector<float> max_val(N, -std::numeric_limits<float>::infinity());
        for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            max_val[i] = std::max(max_val[i], s[i * N + j]);
        // P = exp(S - max), sum
        std::vector<float> sum_exp(N, 0.0f);
        for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
        {
            s[i * N + j] = std::exp(s[i * N + j] - max_val[i]);
            sum_exp[i] += s[i * N + j];
        }
        // softmax normalize
        for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            s[i * N + j] /= sum_exp[i];
        // O = P V
        for(int i = 0; i < N; ++i)
        for(int d = 0; d < D; ++d)
        {
            float acc = 0.0f;
            for(int j = 0; j < N; ++j)
                acc += s[i * N + j] * v_ptr[j * D + d];
            o_ptr[i * D + d] = acc;
        }
    }
}