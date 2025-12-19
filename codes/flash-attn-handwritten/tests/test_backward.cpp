#include <gtest/gtest.h>
#include "flash_attn.h"
#include <vector>
#include <cuda_runtime.h>

void rand_vec(std::vector<float>& v) { for(auto& x : v) x = rand()/float(RAND_MAX); }

TEST(FlashBackward, CpuCudaAgree) {
    const int B = 2, H = 4, N = 256, D = 64;
    size_t len = B*H*N*D;
    std::vector<float> q(len), k(len), v(len), dout(len);
    rand_vec(q); rand_vec(k); rand_vec(v); rand_vec(dout);

    // 先跑前向拿 out
    std::vector<float> out(len);
    flash_attn_v2_fwd(q.data(),k.data(),v.data(),out.data(), B,H,N,D, Arch::V2_CPU);

    // CPU 反向
    FlashAttnGrad g_cpu = flash_attn_v2_backward(
        q.data(),k.data(),v.data(),out.data(),dout.data(), B,H,N,D, Arch::V2_CPU);

    // CUDA 反向
    float *dq, *dk, *dv;
    cudaMalloc(&dq, len*sizeof(float));
    cudaMalloc(&dk, len*sizeof(float));
    cudaMalloc(&dv, len*sizeof(float));
    float *dq_out, *dk_out, *dv_out;
    cudaMalloc(&dq_out, len*sizeof(float));
    cudaMalloc(&dk_out, len*sizeof(float));
    cudaMalloc(&dv_out, len*sizeof(float));

    cudaMemcpy(dq, q.data(), len*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k.data(), len*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v.data(), len*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dq_out, dout.data(), len*sizeof(float), cudaMemcpyHostToDevice);

    // 前向 cuda
    flash_attn_v2_fwd(dq,dk,dv,dq_out, B,H,N,D, Arch::V2_CUDA);
    // 反向 cuda
    FlashAttnGrad g_cuda = flash_attn_v2_backward(
        dq,dk,dv,dq_out,dq_out, B,H,N,D, Arch::V2_CUDA);

    // 下载对比
    std::vector<float> gq_cpu(len), gk_cpu(len), gv_cpu(len);
    std::vector<float> gq_cuda(len), gk_cuda(len), gv_cuda(len);
    cudaMemcpy(gq_cuda.data(), g_cuda.dQ, len*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gk_cuda.data(), g_cuda.dK, len*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gv_cuda.data(), g_cuda.dV, len*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gq_cpu.data(), g_cpu.dQ, len*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gk_cpu.data(), g_cpu.dK, len*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gv_cpu.data(), g_cpu.dV, len*sizeof(float), cudaMemcpyDeviceToHost);

    double diff = 0;
    for(size_t i=0;i<len;++i){
        diff = std::max(diff, std::fabs(gq_cpu[i]-gq_cuda[i]));
        diff = std::max(diff, std::fabs(gk_cpu[i]-gk_cuda[i]));
        diff = std::max(diff, std::fabs(gv_cpu[i]-gv_cuda[i]));
    }
    printf("V2 backward max diff = %e\n", diff);
    EXPECT_LT(diff, 1e-4);

    cudaFree(dq); cudaFree(dk); cudaFree(dv);
    cudaFree(dq_out); cudaFree(dk_out); cudaFree(dv_out);
    cudaFree(g_cpu.dQ); cudaFree(g_cpu.dK); cudaFree(g_cpu.dV);
}