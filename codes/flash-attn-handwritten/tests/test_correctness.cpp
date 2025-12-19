#include <gtest/gtest.h>
#include "flash_attn.h"
#include <vector>
#include <cuda_runtime.h>
#include <math.h>

void rand_vec(std::vector<float>& v) { for(auto& x : v) x = rand()/float(RAND_MAX); }

TEST(FlashTest, V2_CpuCudaAgree) {
    const int B = 2, H = 4, N = 256, D = 64;
    size_t len = B*H*N*D;
    std::vector<float> q(len), k(len), v(len);
    rand_vec(q); rand_vec(k); rand_vec(v);
    std::vector<float> o_cpu(len), o_cuda(len);

    flash_attn_v2_fwd(q.data(),k.data(),v.data(),o_cpu.data(), B,H,N,D, Arch::V2_CPU);

    float *dq,*dk,*dv,*do_;
    cudaMalloc(&dq,len*sizeof(float));
    cudaMalloc(&dk,len*sizeof(float));
    cudaMalloc(&dv,len*sizeof(float));
    cudaMalloc(&do_,len*sizeof(float));
    cudaMemcpy(dq,q.data(),len*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dk,k.data(),len*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dv,v.data(),len*sizeof(float),cudaMemcpyHostToDevice);

    flash_attn_v2_fwd(dq,dk,dv,do_, B,H,N,D, Arch::V2_CUDA);
    cudaMemcpy(o_cuda.data(),do_,len*sizeof(float),cudaMemcpyDeviceToHost);

    float max_diff = 0;
    for(size_t i=0;i<len;++i) max_diff = std::max(max_diff, std::fabs(o_cpu[i]-o_cuda[i]));
    printf("V2 max diff = %e\n", max_diff);
    EXPECT_LT(max_diff, 1e-4);

    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(do_);
}