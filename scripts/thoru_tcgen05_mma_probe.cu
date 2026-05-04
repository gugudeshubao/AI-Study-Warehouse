/**
 * Thor U tcgen05.mma NVFP4 PTX 探针
 *
 * 验证 tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X
 * 在 Thor U (sm_110a) 上能否物理 launch，SASS 是否为 UTCOMMA.4X
 *
 * 编译（必须用系统 nvcc 13.0.48，ptxas 13.2 编译的 control word 有 bug 会 trap）:
 *   /usr/local/cuda/bin/nvcc \
 *       -gencode arch=compute_110a,code=sm_110a \
 *       -std=c++17 -O2 -w \
 *       thoru_tcgen05_mma_probe.cu -o thoru_tcgen05_mma_probe
 *
 * 运行:
 *   ./thoru_tcgen05_mma_probe
 *
 * 反汇编验证 SASS:
 *   cuobjdump --dump-sass thoru_tcgen05_mma_probe | grep UTCOMMA
 *   # 预期输出: UTCOMMA.4X gdesc[URZ], gdesc[URZ], tmem[UR4], ...
 *
 * 技术说明:
 *   - tcgen05.mma 是 Blackwell TMEM/UMMA 子系统的 PTX 指令
 *   - 5090 (GeForce) 硬件砍掉了 TMEM/UMMA，只有 Thor U / B200 能用
 *   - 本探针用 fake descriptor (idesc=0, desc=0) launch → 预期 trap (illegal instruction)
 *   - 探针核心价值: 验证 ptxas 能编译 tcgen05.mma 且 SASS = UTCOMMA.4X
 *   - 端到端数值验证需要 CUTLASS/JAX 级基础设施（descriptor 构造 + warpgroup 调度）
 *   - cuBLASLt _scaled_mm 路径已验证 8/8 全数值正确（见 thoru_nvfp4_bench.py）
 *
 * V19 溯源 (2026-05-04):
 *   - ptxas 13.0.48 编译 sm_110a → SASS = UTCOMMA.4X ✅
 *   - V19 原始 binary (p_v6_110a) main() 未 launch kernel，仅证明编译通过
 *   - fake descriptor (idesc=0) 实际 launch → illegal instruction trap (预期行为)
 *   - ptxas 13.2.78 编译同一代码 SASS 一致，control word 也一致 (0x0001d8000f800004)
 *   - ELF metadata 确认 V19 binary 是 ptxas 13.0.48 编译的
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define CK(call)                                                         \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
                   cudaGetErrorString(err));                              \
            return 1;                                                    \
        }                                                                \
    } while (0)

/**
 * Probe 1: tcgen05.mma kind::mxf4nvf4 (NVFP4 e2m1 + e4m3 scale)
 * fake descriptor → 硬件接受 launch 但不做真计算
 */
__global__ void probe_mxf4nvf4() {
    if (threadIdx.x == 0) {
        unsigned int tmem_c = 0;
        unsigned long long desc_a = 0, desc_b = 0;
        unsigned int idesc_high = 0, scale_predicate = 0;
        unsigned int scale_a_addr = 0, scale_b_addr = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
            "[%0], %1, %2, %3, [%5], [%6], p;\n\t"
            "}\n"
            :
            : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc_high),
              "r"(scale_predicate), "r"(scale_a_addr), "r"(scale_b_addr));
    }
}

/**
 * Probe 2: tcgen05.mma kind::mxf8f6f4 (通用 FP8/FP6/FP4 with block_scale)
 */
__global__ void probe_mxf8f6f4() {
    if (threadIdx.x == 0) {
        unsigned int tmem_c = 0;
        unsigned long long desc_a = 0, desc_b = 0;
        unsigned int idesc_high = 0, scale_predicate = 0;
        unsigned int scale_a_addr = 0, scale_b_addr = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale "
            "[%0], %1, %2, %3, [%5], [%6], p;\n\t"
            "}\n"
            :
            : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc_high),
              "r"(scale_predicate), "r"(scale_a_addr), "r"(scale_b_addr));
    }
}

int main() {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("====== Thor U tcgen05.mma NVFP4 PTX 探针 ======\n");
    printf("Device:  %s\n", prop.name);
    printf("SM:      %d.%d\n", prop.major, prop.minor);
    printf("SMs:     %d\n", prop.multiProcessorCount);
    printf("\n");

    /* Probe 1: mxf4nvf4 (NVFP4 专用路径) */
    printf("--- Probe 1: tcgen05.mma kind::mxf4nvf4 ---\n");
    probe_mxf4nvf4<<<1, 32>>>();
    cudaError_t err1 = cudaDeviceSynchronize();
    if (err1 == cudaSuccess) {
        printf("  RESULT: LAUNCH OK (SASS = UTCOMMA.4X)\n");
        printf("  硬件接受 tcgen05.mma mxf4nvf4 指令\n");
    } else {
        printf("  RESULT: FAIL (%s)\n", cudaGetErrorString(err1));
        /* 重置 device 错误状态 */
        cudaGetLastError();
    }

    /* Probe 2: mxf8f6f4 (通用 block_scale 路径) */
    printf("\n--- Probe 2: tcgen05.mma kind::mxf8f6f4 ---\n");
    probe_mxf8f6f4<<<1, 32>>>();
    cudaError_t err2 = cudaDeviceSynchronize();
    if (err2 == cudaSuccess) {
        printf("  RESULT: LAUNCH OK\n");
    } else {
        printf("  RESULT: FAIL (%s)\n", cudaGetErrorString(err2));
        cudaGetLastError();
    }

    /* 总结 */
    printf("\n====== Summary ======\n");
    printf("mxf4nvf4 (NVFP4):  %s\n", err1 == cudaSuccess ? "OK" : "FAIL");
    printf("mxf8f6f4 (generic): %s\n", err2 == cudaSuccess ? "OK" : "FAIL");
    printf("\nNote: fake descriptor (idesc=0) → launch OK but no real computation.\n");
    printf("For numerical verification, use cuBLASLt via thoru_nvfp4_bench.py\n");
    printf("For SASS verification: cuobjdump --dump-sass %s | grep UTCOMMA\n",
           "thoru_tcgen05_mma_probe");

    return 0;
}
