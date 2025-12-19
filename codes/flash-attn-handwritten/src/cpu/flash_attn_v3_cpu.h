#pragma once
#include <cstddef>

void flash_attn_v3_cpu_fwd(const float* q, const float* k, const float* v,
                           float* out,
                           int B, int H, int N, int D);
                           #include "flash_attn.h"

FlashAttnGrad flash_attn_v3_cpu_backward(const float* q, const float* k, const float* v,
                                         const float* out, const float* dout,
                                         int B, int H, int N, int D);