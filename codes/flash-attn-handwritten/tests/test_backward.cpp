#include <gtest/gtest.h>
#include "flash_attn.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

void rand_vec(std::vector<float>& v) { for(auto& x : v) x = rand()/float(RAND_MAX); }

TEST(FlashBackward, CpuCudaAgree) {
    const int B = 2, H = 4, N = 16, D = 64;
    size_t len = B*H*N*D;
    std::vector<float> q(len), k(len), v(len), dout(len);
    rand_vec(q); rand_vec(k); rand_vec(v); rand_vec(dout);

    // 先跑前向拿 out
    std::vector<float> out(len);
    flash_attn_v2_fwd(q.data(),k.data(),v.data(),out.data(), B,H,N,D, Arch::V2_CPU);

    // CPU 反向
    FlashAttnGrad g_cpu = flash_attn_v2_backward(
        q.data(),k.data(),v.data(),out.data(),dout.data(), B,H,N,D, Arch::V2_CPU);

    float *dq, *dk, *dv, *dout_dev, *out_dev;
    cudaMalloc(&dq, len*sizeof(float));
    cudaMalloc(&dk, len*sizeof(float));
    cudaMalloc(&dv, len*sizeof(float));
    cudaMalloc(&dout_dev, len*sizeof(float));
    cudaMalloc(&out_dev, len*sizeof(float));

    std::vector<float> gq_cpu(len), gk_cpu(len), gv_cpu(len);
    std::memcpy(gq_cpu.data(), g_cpu.dQ, len*sizeof(float));
    std::memcpy(gk_cpu.data(), g_cpu.dK, len*sizeof(float));
    std::memcpy(gv_cpu.data(), g_cpu.dV, len*sizeof(float));

    auto run_and_compare = [&](Arch arch, const char* label) {
        cudaMemcpy(dq, q.data(), len*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dk, k.data(), len*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dv, v.data(), len*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dout_dev, dout.data(), len*sizeof(float), cudaMemcpyHostToDevice);

        flash_attn_v2_fwd(dq, dk, dv, out_dev, B, H, N, D, arch);
        FlashAttnGrad g_dev = flash_attn_v2_backward(dq, dk, dv, out_dev, dout_dev, B, H, N, D, arch);

        std::vector<float> gq_dev(len), gk_dev(len), gv_dev(len);
        cudaMemcpy(gq_dev.data(), g_dev.dQ, len*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gk_dev.data(), g_dev.dK, len*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gv_dev.data(), g_dev.dV, len*sizeof(float), cudaMemcpyDeviceToHost);

        double diff = 0;
        for(size_t i = 0; i < len; ++i){
            diff = std::max(diff, static_cast<double>(std::fabs(gq_cpu[i] - gq_dev[i])));
            diff = std::max(diff, static_cast<double>(std::fabs(gk_cpu[i] - gk_dev[i])));
            diff = std::max(diff, static_cast<double>(std::fabs(gv_cpu[i] - gv_dev[i])));
        }
        printf("%s backward max diff = %e\n", label, diff);
        EXPECT_LT(diff, 1e-4);

        cudaFree(g_dev.dQ);
        cudaFree(g_dev.dK);
        cudaFree(g_dev.dV);
    };

    run_and_compare(Arch::V2_CUDA, "V2_CUDA");
    run_and_compare(Arch::V2_CUBLAS, "V2_CUBLAS");

    cudaFree(dq); cudaFree(dk); cudaFree(dv);
    cudaFree(dout_dev); cudaFree(out_dev);
    cudaFreeHost(g_cpu.dQ); cudaFreeHost(g_cpu.dK); cudaFreeHost(g_cpu.dV);
}

TEST(FlashBackward, V3CudaAgreesWithCpu) {
    const int B = 2, H = 4, N = 128, D = 64;
    size_t len = B * H * N * D;
    std::vector<float> q(len), k(len), v(len), dout(len);
    rand_vec(q); rand_vec(k); rand_vec(v); rand_vec(dout);

    std::vector<float> out_cpu(len);
    flash_attn_v2_fwd(q.data(), k.data(), v.data(), out_cpu.data(), B, H, N, D, Arch::V2_CPU);

    FlashAttnGrad g_cpu = flash_attn_v2_backward(
        q.data(), k.data(), v.data(), out_cpu.data(), dout.data(), B, H, N, D, Arch::V2_CPU);

    float *dq, *dk, *dv, *dout_dev, *out_dev;
    cudaMalloc(&dq, len * sizeof(float));
    cudaMalloc(&dk, len * sizeof(float));
    cudaMalloc(&dv, len * sizeof(float));
    cudaMalloc(&dout_dev, len * sizeof(float));
    cudaMalloc(&out_dev, len * sizeof(float));

    cudaMemcpy(dq, q.data(), len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k.data(), len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v.data(), len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dout_dev, dout.data(), len * sizeof(float), cudaMemcpyHostToDevice);

    flash_attn_v2_fwd(dq, dk, dv, out_dev, B, H, N, D, Arch::V3_CUDA);
    FlashAttnGrad g_v3 = flash_attn_v2_backward(
        dq, dk, dv, out_dev, dout_dev, B, H, N, D, Arch::V3_CUDA);

    std::vector<float> gq_cpu(len), gk_cpu(len), gv_cpu(len);
    std::vector<float> gq_v3(len), gk_v3(len), gv_v3(len);
    std::memcpy(gq_cpu.data(), g_cpu.dQ, len * sizeof(float));
    std::memcpy(gk_cpu.data(), g_cpu.dK, len * sizeof(float));
    std::memcpy(gv_cpu.data(), g_cpu.dV, len * sizeof(float));
    cudaMemcpy(gq_v3.data(), g_v3.dQ, len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gk_v3.data(), g_v3.dK, len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gv_v3.data(), g_v3.dV, len * sizeof(float), cudaMemcpyDeviceToHost);

    double diff = 0.0;
    double diff_q = 0.0;
    double diff_k = 0.0;
    double diff_v = 0.0;
    for (size_t i = 0; i < len; ++i) {
        diff_q = std::max(diff_q, static_cast<double>(std::fabs(gq_cpu[i] - gq_v3[i])));
        diff_k = std::max(diff_k, static_cast<double>(std::fabs(gk_cpu[i] - gk_v3[i])));
        diff_v = std::max(diff_v, static_cast<double>(std::fabs(gv_cpu[i] - gv_v3[i])));
    }
    diff = std::max(diff_q, std::max(diff_k, diff_v));
    printf("V3_CUDA backward diff_q = %e, diff_k = %e, diff_v = %e\n", diff_q, diff_k, diff_v);
    printf("V3_CUDA backward max diff = %e\n", diff);
    EXPECT_LT(diff, 1e-4);

    cudaFree(dq); cudaFree(dk); cudaFree(dv);
    cudaFree(dout_dev); cudaFree(out_dev);
    cudaFree(g_v3.dQ); cudaFree(g_v3.dK); cudaFree(g_v3.dV);
    cudaFreeHost(g_cpu.dQ); cudaFreeHost(g_cpu.dK); cudaFreeHost(g_cpu.dV);
}
