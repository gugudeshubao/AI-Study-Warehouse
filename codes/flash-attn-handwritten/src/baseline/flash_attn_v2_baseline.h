#pragma once
#include <cstddef>

// 对外暴露的 V2 cuBLAS 基线接口
void flash_attn_v2_cublas_fwd(const float *q, const float *k, const float *v,
                              float *out,
                              int B, int H, int N, int D);
#include "flash_attn.h" // FlashAttnGrad

FlashAttnGrad cublas_flash_v2_backward(const float* q, const float* k, const float* v,
                                       const float* out, const float* dout,
                                       int B, int H, int N, int D);