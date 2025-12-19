#pragma once
#include <cuda_fp8.h>

void flash_attn_v2_blackwell_tcgen05_fwd(const __fp8_e4m3* q, const __fp8_e4m3* k, const __fp8_e4m3* v,
                                         float* out,
                                         int B, int H, int N, int D);

#include "flash_attn.h" // FlashAttnGrad


FlashAttnGrad flash_attn_v2_blackwell_tcgen05_backward(const __fp8_e4m3* q, const __fp8_e4m3* k, const __fp8_e4m3* v,
                                                       const float* out, const float* dout,
                                                       int B, int H, int N, int D);