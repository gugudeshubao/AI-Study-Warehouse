#pragma once
#include <cstddef>

// 主机端统一接口
void flash_attn_v2_hopper_wgmma_fwd(const float* q, const float* k, const float* v,
                                    float* out,
                                    int B, int H, int N, int D);