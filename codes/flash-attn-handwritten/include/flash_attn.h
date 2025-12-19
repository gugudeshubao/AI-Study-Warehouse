#pragma once
#include <cstddef>

enum class Arch
{
    V2_CPU = 0,
    V2_CUDA,
    V2_AMPERE_WMMA,
    V2_AMPERE_MMA,
    V2_HOPPER_WMMA,
    V2_HOPPER_MMA,
    V2_BLACKWELL_WMMA,
    V2_BLACKWELL_MMA,
    V2_CUBLAS,
    V2_HOPPER_WGMMA,      // 新增
    V2_BLACKWELL_TCGEN05, // 新增
    V3_CUDA,
    V4_CUDA
};

void flash_attn_v2_fwd(const float *q, const float *k, const float *v,
                       float *out,
                       int B, int H, int N, int D,
                       Arch arch);

/* 将来继续加
void flash_attn_v3_fwd(...);
void flash_attn_v4_fwd(...);
*/


// 反向接口：dout 是前向输出梯度，返回 {dQ,dK,dV}
struct FlashAttnGrad {
    float* dQ;
    float* dK;
    float* dV;
};

FlashAttnGrad flash_attn_v2_backward(const float* q, const float* k, const float* v,
                                     const float* out, const float* dout,
                                     int B, int H, int N, int D,
                                     Arch arch);



//                                      // V3 前向
// void flash_attn_v3_fwd(const float* q, const float* k, const float* v,
//                        float* out,
//                        int B, int H, int N, int D,
//                        Arch arch);

// // V3 反向
// struct FlashAttnGrad {
//     float* dQ;
//     float* dK;
//     float* dV;
// };
// FlashAttnGrad flash_attn_v3_backward(const float* q, const float* k, const float* v,
//                                      const float* out, const float* dout,
//                                      int B, int H, int N, int D,
//                                      Arch arch);