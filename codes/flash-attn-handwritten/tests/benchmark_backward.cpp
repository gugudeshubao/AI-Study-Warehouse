#include "flash_attn.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static const char* arch_name(Arch arch)
{
    switch (arch)
    {
    case Arch::V2_CUBLAS: return "V2_CUBLAS";
    case Arch::V3_CUDA: return "V3_CUDA";
    default: return "UNKNOWN";
    }
}

int main(int argc, char* argv[])
{
    if (argc < 6) {
        fprintf(stderr, "usage: %s B H N D arch [iters] [warmup]\n", argv[0]);
        return 1;
    }

    const int B = atoi(argv[1]);
    const int H = atoi(argv[2]);
    const int N = atoi(argv[3]);
    const int D = atoi(argv[4]);
    const int arch_i = atoi(argv[5]);
    const int iters = argc >= 7 ? atoi(argv[6]) : 50;
    const int warmup = argc >= 8 ? atoi(argv[7]) : 10;
    const Arch arch = static_cast<Arch>(arch_i);

    size_t len = static_cast<size_t>(B) * H * N * D;
    std::vector<float> q(len), k(len), v(len), dout(len);
    for (size_t i = 0; i < len; ++i) {
        q[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        k[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        v[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        dout[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }

    float *dq, *dk, *dv, *dout_dev, *out_dev;
    cudaMalloc(&dq, len * sizeof(float));
    cudaMalloc(&dk, len * sizeof(float));
    cudaMalloc(&dv, len * sizeof(float));
    cudaMalloc(&dout_dev, len * sizeof(float));
    cudaMalloc(&out_dev, len * sizeof(float));
    cudaMemcpy(dq, q.data(), len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k.data(), len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v.data(), len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dout_dev, dout.data(), len * sizeof(float), cudaMemcpyHostToDevice);

    flash_attn_v2_fwd(dq, dk, dv, out_dev, B, H, N, D, arch);
    cudaDeviceSynchronize();

    auto run_once = [&]() {
        FlashAttnGrad grads = flash_attn_v2_backward(dq, dk, dv, out_dev, dout_dev, B, H, N, D, arch);
        cudaFree(grads.dQ);
        cudaFree(grads.dK);
        cudaFree(grads.dV);
    };

    for (int i = 0; i < warmup; ++i)
        run_once();
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
        run_once();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= iters;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const double flops = 8.0 * B * H * N * N * D;
    const double tflops = flops / ms / 1e9;

    printf("{\"b\":%d,\"h\":%d,\"n\":%d,\"d\":%d,\"arch\":%d,\"arch_name\":\"%s\",\"iters\":%d,\"warmup\":%d,\"ms\":%.3f,\"tflops\":%.2f}\n",
           B, H, N, D, arch_i, arch_name(arch), iters, warmup, ms, tflops);

    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(dout_dev);
    cudaFree(out_dev);
    return 0;
}
