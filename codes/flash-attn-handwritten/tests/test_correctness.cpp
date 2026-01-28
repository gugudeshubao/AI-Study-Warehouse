#include <gtest/gtest.h>
#include "flash_attn.h"
#include <vector>
#include <cuda_runtime.h>
#include <math.h>

void rand_vec(std::vector<float> &v)
{
    for (auto &x : v)
        x = rand() / float(RAND_MAX);
}

TEST(FlashTest, V2_CpuCudaAgree)
{
    // const int B = 2, H = 4, N = 256, D = 64;
    const int B = 2, H = 4, N = 16, D = 64;
    size_t len = B * H * N * D;
    std::vector<float> q(len), k(len), v(len);
    rand_vec(q);
    rand_vec(k);
    rand_vec(v);
    std::vector<float> o_cpu(len), o_cuda(len);


    float max_diff = 0;
    float *dq, *dk, *dv, *do_;
    cudaMalloc(&dq, len * sizeof(float));
    cudaMalloc(&dk, len * sizeof(float));
    cudaMalloc(&dv, len * sizeof(float));
    cudaMalloc(&do_, len * sizeof(float));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    size_t share_mem_hardLimit = prop.sharedMemPerBlock; // 48 KB default

    printf("share mem hardlimit=%d,totalGlobalMem(GB)=%ld,regsPerBlock=%d,warpSize=%d,maxThreadsPerBlock=%d,maxThreadsPerMultiprocessor=%d\n",
           share_mem_hardLimit, prop.totalGlobalMem / 1024 / 1024 / 1024, prop.regsPerBlock, prop.warpSize, prop.maxThreadsPerBlock, prop.maxThreadsPerMultiProcessor);
    printf("GPU Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SM count: %d\n", prop.multiProcessorCount);

    /* ---------- CUDA Core ---------- */
    int coresPerSM = 0;
    if (prop.major == 7 && prop.minor == 0)
        coresPerSM = 64; // Volta
    if (prop.major == 7 && prop.minor == 5)
        coresPerSM = 64; // Turing
    if (prop.major == 8 && prop.minor == 0)
        coresPerSM = 64; // Ampere GA100
    if (prop.major == 8 && prop.minor >= 6)
        coresPerSM = 128; // Ampere GA10x / Ada
    if (prop.major == 9)
        coresPerSM = 128; // Hopper

    int totalCores = prop.multiProcessorCount * coresPerSM;
    printf("CUDA Cores per SM: %d\n", coresPerSM);
    printf("Total CUDA Cores: %d\n", totalCores);

    /* ---------- Tensor Core ---------- */
    int tcPerSM = 0;
    if (prop.major == 7 && prop.minor == 0)
        tcPerSM = 2;
    if (prop.major == 7 && prop.minor == 5)
        tcPerSM = 1;
    if (prop.major == 8)
        tcPerSM = 1;
    if (prop.major == 9)
        tcPerSM = 1;

    if (tcPerSM > 0)
    {
        printf("Tensor Cores per SM: %d\n", tcPerSM);
        printf("Total Tensor Cores: %d\n", prop.multiProcessorCount * tcPerSM);
    }
    else
    {
        printf("No Tensor Cores on this architecture.\n");
    }

    // main 里最先调用
    cudaError_t err = cudaSetDevice(0); // 0 就是 nvidia-smi 里看到的 GPU-ID
    if (err != cudaSuccess)
    {
        printf("cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    printf("test Arch::V2_CPU\n");
    // cpu
    flash_attn_v2_fwd(q.data(), k.data(), v.data(), o_cpu.data(), B, H, N, D, Arch::V2_CPU);

    // cublas core
    printf("test Arch::V2_CUBLAS\n");
    cudaMemcpy(dq, q.data(), len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k.data(), len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v.data(), len * sizeof(float), cudaMemcpyHostToDevice);

    flash_attn_v2_fwd(dq, dk, dv, do_, B, H, N, D, Arch::V2_CUBLAS);
    cudaMemcpy(o_cuda.data(), do_, len * sizeof(float), cudaMemcpyDeviceToHost);
    max_diff = 0;
    for (size_t i = 0; i < len; ++i)
        max_diff = std::max(max_diff, std::fabs(o_cpu[i] - o_cuda[i]));
    printf("V2 max diff = %e\n", max_diff);
    EXPECT_LT(max_diff, 1e-4);

    // cuda core
    cudaMemcpy(dq, q.data(), len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k.data(), len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v.data(), len * sizeof(float), cudaMemcpyHostToDevice);
    printf("test Arch::V2_CUDA\n");
    flash_attn_v2_fwd(dq, dk, dv, do_, B, H, N, D, Arch::V2_CUDA);
    cudaMemcpy(o_cuda.data(), do_, len * sizeof(float), cudaMemcpyDeviceToHost);
    max_diff = 0;
    for (size_t i = 0; i < len; ++i)
        max_diff = std::max(max_diff, std::fabs(o_cpu[i] - o_cuda[i]));
    printf("V2 max diff = %e\n", max_diff);
    EXPECT_LT(max_diff, 1e-4);
    // ampere mma
    cudaMemcpy(dq, q.data(), len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k.data(), len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v.data(), len * sizeof(float), cudaMemcpyHostToDevice);
    printf("test Arch::V2_AMPERE_MMA\n");
    flash_attn_v2_fwd(dq, dk, dv, do_, B, H, N, D, Arch::V2_AMPERE_MMA);
    cudaMemcpy(o_cuda.data(), do_, len * sizeof(float), cudaMemcpyDeviceToHost);
    max_diff = 0;
    for (size_t i = 0; i < len; ++i)
        max_diff = std::max(max_diff, std::fabs(o_cpu[i] - o_cuda[i]));
    printf("V2 max diff = %e\n", max_diff);
    EXPECT_LT(max_diff, 1e-4);
    // ampere wmma
    cudaMemcpy(dq, q.data(), len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k.data(), len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v.data(), len * sizeof(float), cudaMemcpyHostToDevice);
    printf("test Arch::V2_AMPERE_WMMA\n");
    flash_attn_v2_fwd(dq, dk, dv, do_, B, H, N, D, Arch::V2_AMPERE_WMMA);
    cudaMemcpy(o_cuda.data(), do_, len * sizeof(float), cudaMemcpyDeviceToHost);
    max_diff = 0;
    for (size_t i = 0; i < len; ++i)
        max_diff = std::max(max_diff, std::fabs(o_cpu[i] - o_cuda[i]));
    printf("V2 max diff = %e\n", max_diff);
    EXPECT_LT(max_diff, 1e-4);

    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(do_);
}