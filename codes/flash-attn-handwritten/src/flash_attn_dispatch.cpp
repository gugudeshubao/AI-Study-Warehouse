#include "flash_attn.h"
#include "cpu/flash_attn_v2_cpu.h"
#include "cuda/flash_attn_v2_cuda.cuh"
#include "baseline/flash_attn_v2_baseline.h"

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
    default:             break;   // 后续加 V2_AMPERE 等
    }
}