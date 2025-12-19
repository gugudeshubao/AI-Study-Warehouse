#pragma once
#include <cstddef>

void flash_attn_v2_cpu_fwd(const float* q, const float* k, const float* v,
                        float* out,
                        int B, int H, int N, int D);