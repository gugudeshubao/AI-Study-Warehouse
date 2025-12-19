#include "flash_attn.h"
#include "cpu/flash_attn_v2_cpu.h"
#include "cuda/flash_attn_v2_cuda.cuh"
#include "baseline/flash_attn_v2_baseline.h"
#include "ampere/flash_attn_v2_ampere_wmma.h"
#include "ampere/flash_attn_v2_ampere_mma.h"
// #include "hopper/flash_attn_v2_hopper_wgmma.h"
// #include "blackwell/flash_attn_v2_blackwell_tcgen05.h"


void flash_attn_v2_fwd(const float* q, const float* k, const float* v,
                       float* out,
                       int B, int H, int N, int D,
                       Arch arch)
{
    switch(arch)
    {
    case Arch::V2_CPU:   flash_attn_v2_cpu_fwd(q,k,v,out,B,H,N,D); break;
    case Arch::V2_CUDA:  flash_attn_v2_cuda_fwd(q,k,v,out,B,H,N,D); break;
    case Arch::V2_CUBLAS: flash_attn_v2_cublas_fwd(q,k,v,out,B,H,N,D); break;
    case Arch::V2_AMPERE_WMMA: flash_attn_v2_ampere_wmma_fwd(q,k,v,out,B,H,N,D); break;
    // case Arch::V2_AMPERE_MMA: flash_attn_v2_ampere_mma_fwd(q,k,v,out,B,H,N,D); break;
    // case Arch::V2_HOPPER_WGMMA:     flash_attn_v2_hopper_wgmma_fwd(q,k,v,out,B,H,N,D); break;
    // case Arch::V2_BLACKWELL_TCGEN05:{
    //     // 先做一次 host 端 fp8 转换（实测可再优化为预转换）
    //     size_t len = B*H*N*D;
    //     __fp8_e4m3 *q8, *k8, *v8;
    //     cudaMalloc(&q8, len * sizeof(__fp8_e4m3));
    //     cudaMalloc(&k8, len * sizeof(__fp8_e4m3));
    //     cudaMalloc(&v8, len * sizeof(__fp8_e4m3));
    //     cudaMemcpy(q8, q, len * sizeof(float), cudaMemcpyDeviceToDevice); // 这里简写：实际用 cublasConvert
    //     cudaMemcpy(k8, k, len * sizeof(float), cudaMemcpyDeviceToDevice);
    //     cudaMemcpy(v8, v, len * sizeof(float), cudaMemcpyDeviceToDevice);
    //     flash_attn_v2_blackwell_tcgen05_fwd(q8, k8, v8, out, B, H, N, D);
    //     cudaFree(q8); cudaFree(k8); cudaFree(v8);
    //     break;
    // }
    default:             break;   // 后续加 V2_AMPERE 等
    }
}