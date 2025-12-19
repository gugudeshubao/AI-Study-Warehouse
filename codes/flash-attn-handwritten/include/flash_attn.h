#pragma once
#include <cstddef>

enum class Arch { V2_CPU = 0, V2_CUDA, V2_AMPERE, V2_HOPPER, V2_BLACKWELL,V2_CUBLAS,
                  V3_CUDA, V4_CUDA };

void flash_attn_v2_fwd(const float* q, const float* k, const float* v,
                       float* out,
                       int B, int H, int N, int D,
                       Arch arch);

/* 将来继续加
void flash_attn_v3_fwd(...);
void flash_attn_v4_fwd(...);
*/