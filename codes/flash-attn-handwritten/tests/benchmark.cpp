#include "flash_attn.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static cublasHandle_t get_cublas_handle()
{
    static cublasHandle_t h = []{
        cublasHandle_t tmp; cublasCreate(&tmp); return tmp;
    }();
    return h;
}

// 对外暴露的统一 cuBLAS 基线（V2）
void flash_attn_v2_cublas_fwd(const float* q, const float* k, const float* v,
                              float* out,
                              int B, int H, int N, int D)
{
    cublasHandle_t handle = get_cublas_handle();
    const float alpha = 1.0f, beta = 0.0f;
    size_t strideQD = N * D;

    // 1) S = QK^T   (B*H 个 N×N)
    float* s;
    cudaMalloc(&s, B*H*N*N*sizeof(float));
    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, N, D,
        &alpha,
        k, CUDA_R_32F, D, strideQD,
        q, CUDA_R_32F, D, strideQD,
        &beta,
        s, CUDA_R_32F, N, N*N,
        B*H,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // 2) row-wise Softmax（手写 CUDA kernel，<0.1 ms）
    auto softmax_inplace = [](float* s, int BH, int N){
        dim3 block(128);
        dim3 grid((BH*N+block.x-1)/block.x);
        // 这里你可以直接复用之前写的简易 softmax kernel
        // 为了单文件可读，先 inline 一个极简实现
        auto kernel = [] __global__ (float* s, int N){
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= N*gridDim.x) return;
            int row = idx / N;
            int i   = idx % N;
            const float* rowPtr = s + row*N;
            // max
            float max_val = -1e38f;
            for(int j=0;j<N;++j) max_val = fmaxf(max_val, rowPtr[j]);
            // exp & sum
            float sum = 0.0f;
            for(int j=0;j<N;++j){
                float ev = expf(rowPtr[j] - max_val);
                rowPtr[j] = ev;
                sum += ev;
            }
            // normalize
            for(int j=0;j<N;++j) rowPtr[j] /= sum;
        };
        kernel<<<grid,block>>>(s, N);
        cudaDeviceSynchronize();
    };
    softmax_inplace(s, B*H, N);

    // 3) O = S V
    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        D, N, N,
        &alpha,
        v, CUDA_R_32F, D, strideQD,
        s, CUDA_R_32F, N, N*N,
        &beta,
        out, CUDA_R_32F, D, strideQD,
        B*H,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cudaFree(s);
}

int main(int argc, char* argv[])
{
    if(argc < 6){
        fprintf(stderr, "usage: %s B H N D arch\n"
                        "arch: 0=V2_CPU 1=V2_CUDA 2=V2_CUBLAS\n", argv[0]);
        return 1;
    }
    int B  = atoi(argv[1]);
    int H  = atoi(argv[2]);
    int N  = atoi(argv[3]);
    int D  = atoi(argv[4]);
    int arch_i = atoi(argv[5]);
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

    // warm-up
    if(arch == Arch::V2_CPU)
        flash_attn_v2_fwd(q.data(),k.data(),v.data(),q.data(),B,H,N,D, arch);
    else
        flash_attn_v2_fwd(dq,dk,dv,do_, B,H,N,D, arch);
    cudaDeviceSynchronize();

    // timing
    auto t0 = std::chrono::steady_clock::now();
    if(arch == Arch::V2_CPU){
        flash_attn_v2_fwd(q.data(),k.data(),v.data(),q.data(),B,H,N,D, arch);
    }else if(arch == Arch::V2_CUDA){
        flash_attn_v2_fwd(dq,dk,dv,do_, B,H,N,D, arch);
    }else if(arch == Arch::V2_CUBLAS){
        flash_attn_v2_cublas_fwd(dq,dk,dv,do_, B,H,N,D);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();

    double ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
    double flops = 4.0 * B * H * N * N * D;      // QK^T + Softmax + PV
    double tflops = flops / ms / 1e9;

    // 输出 json 一行，方便脚本解析
    printf("{\"b\":%d,\"h\":%d,\"n\":%d,\"d\":%d,\"arch\":%d,\"ms\":%.3f,\"tflops\":%.2f}\n",
           B,H,N,D, arch_i, ms, tflops);

    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(do_);
    return 0;
}