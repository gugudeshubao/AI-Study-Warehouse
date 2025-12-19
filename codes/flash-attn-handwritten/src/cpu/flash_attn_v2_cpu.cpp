#include "cpu/flash_attn_v2_cpu.h"
#include <cmath>
#include <vector>
#include "flash_attn.h"
#include <cuda_runtime.h>  // 仅为了使用 cudaMallocHost / cudaFree ,这个后续可以去掉
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


// 辅助：行方向 softmax 的反向
static void softmax_backward_row(float* grad, const float* out, int n) {
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) sum += grad[j] * out[j];
    for (int j = 0; j < n; ++j) grad[j] = out[j] * (grad[j] - sum);
}

FlashAttnGrad flash_attn_v2_cpu_backward(const float* q, const float* k, const float* v,
                                         const float* out, const float* dout,
                                         int B, int H, int N, int D) {
    size_t len = B * H * N * D;
    float *dQ, *dK, *dV;
    cudaMallocHost(&dQ, len * sizeof(float));
    cudaMallocHost(&dK, len * sizeof(float));
    cudaMallocHost(&dV, len * sizeof(float));
    std::fill(dQ, dQ + len, 0.0f);
    std::fill(dK, dK + len, 0.0f);
    std::fill(dV, dV + len, 0.0f);

    int stride = N * D;
    std::vector<float> s(N * N), p(N * N), grad_s(N * N);

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

            // 1) 前向 S = QK^T
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (int d = 0; d < D; ++d)
                        sum += q_ptr[i * D + d] * k_ptr[j * D + d];
                    s[i * N + j] = sum;
                }

            // 2) P = softmax(S)
            for (int i = 0; i < N; ++i) {
                float maxVal = -1e38f;
                for (int j = 0; j < N; ++j) maxVal = fmaxf(maxVal, s[i * N + j]);
                float sum = 0.0f;
                for (int j = 0; j < N; ++j) {
                    p[i * N + j] = expf(s[i * N + j] - maxVal);
                    sum += p[i * N + j];
                }
                for (int j = 0; j < N; ++j) p[i * N + j] /= sum;
            }

            // 3) dV = P^T @ dout
            for (int j = 0; j < N; ++j)
                for (int d = 0; d < D; ++d) {
                    float sum = 0.0f;
                    for (int i = 0; i < N; ++i)
                        sum += p[i * N + j] * dout_ptr[i * D + d];
                    dv_ptr[j * D + d] = sum;
                }

            // 4) grad_S = dout @ V^T
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (int d = 0; d < D; ++d)
                        sum += dout_ptr[i * D + d] * v_ptr[j * D + d];
                    grad_s[i * N + j] = sum;
                }

            // 5) softmax 反向
            for (int i = 0; i < N; ++i)
                softmax_backward_row(&grad_s[i * N], &p[i * N], N);

            // 6) dQ = grad_S @ K
            for (int i = 0; i < N; ++i)
                for (int d = 0; d < D; ++d) {
                    float sum = 0.0f;
                    for (int j = 0; j < N; ++j)
                        sum += grad_s[i * N + j] * k_ptr[j * D + d];
                    dq_ptr[i * D + d] = sum;
                }

            // 7) dK = grad_S^T @ Q
            for (int j = 0; j < N; ++j)
                for (int d = 0; d < D; ++d) {
                    float sum = 0.0f;
                    for (int i = 0; i < N; ++i)
                        sum += grad_s[i * N + j] * q_ptr[i * D + d];
                    dk_ptr[j * D + d] = sum;
                }
        }
    }
     return {dQ, dK, dV};

}