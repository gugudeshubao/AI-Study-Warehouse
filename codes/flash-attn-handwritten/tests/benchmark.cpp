#include "flash_attn.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static const char* arch_name(Arch arch)
{
    switch (arch)
    {
    case Arch::V2_CPU: return "V2_CPU";
    case Arch::V2_CUDA: return "V2_CUDA";
    case Arch::V2_AMPERE_WMMA: return "V2_AMPERE_WMMA";
    case Arch::V2_AMPERE_MMA: return "V2_AMPERE_MMA";
    case Arch::V2_CUBLAS: return "V2_CUBLAS";
    case Arch::V3_CUDA: return "V3_CUDA";
    default: return "UNKNOWN";
    }
}

int main(int argc, char* argv[])
{
    if(argc < 6){
        fprintf(stderr, "usage: %s B H N D arch [iters] [warmup]\n", argv[0]);
        return 1;
    }
    int B  = atoi(argv[1]);
    int H  = atoi(argv[2]);
    int N  = atoi(argv[3]);
    int D  = atoi(argv[4]);
    int arch_i = atoi(argv[5]);
    int iters = argc >= 7 ? atoi(argv[6]) : 50;
    int warmup = argc >= 8 ? atoi(argv[7]) : 10;
    Arch arch = static_cast<Arch>(arch_i);

    size_t len = B*H*N*D;
    std::vector<float> q(len), k(len), v(len);
    for(size_t i=0;i<len;++i){
        q[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        k[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        v[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }

    // device memory
    float *dq, *dk, *dv, *do_;
    cudaMalloc(&dq, len*sizeof(float));
    cudaMalloc(&dk, len*sizeof(float));
    cudaMalloc(&dv, len*sizeof(float));
    cudaMalloc(&do_, len*sizeof(float));
    cudaMemcpy(dq, q.data(), len*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k.data(), len*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v.data(), len*sizeof(float), cudaMemcpyHostToDevice);

    auto run_once = [&]() {
        if (arch == Arch::V2_CPU)
            flash_attn_v2_fwd(q.data(), k.data(), v.data(), q.data(), B, H, N, D, arch);
        else
            flash_attn_v2_fwd(dq, dk, dv, do_, B, H, N, D, arch);
    };

    for (int i = 0; i < warmup; ++i)
        run_once();
    cudaDeviceSynchronize();

    float ms = 0.0f;
    if (arch == Arch::V2_CPU) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
            run_once();
        auto t1 = std::chrono::steady_clock::now();
        ms = static_cast<float>(std::chrono::duration<double, std::milli>(t1 - t0).count() / iters);
    } else {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < iters; ++i)
            run_once();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        ms /= iters;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    double flops = 4.0 * B * H * N * N * D;      // QK^T + Softmax + PV
    double tflops = flops / ms / 1e9;

    // 输出 json 一行，方便脚本解析
    printf("{\"b\":%d,\"h\":%d,\"n\":%d,\"d\":%d,\"arch\":%d,\"arch_name\":\"%s\",\"iters\":%d,\"warmup\":%d,\"ms\":%.3f,\"tflops\":%.2f}\n",
           B,H,N,D, arch_i, arch_name(arch), iters, warmup, ms, tflops);

    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(do_);
    return 0;
}
