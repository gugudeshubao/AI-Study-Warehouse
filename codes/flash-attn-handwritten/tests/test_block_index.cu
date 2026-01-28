#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

constexpr int N = 1 << 10;      // 1024×1024
constexpr int TILE = 32;        // 每个 block 处理 32×32 子块

__global__ void transpose2D(const float* in, float* out, int n) {
    // 共享内存缓存，避免全局内存非合并访问
    __shared__ float tile[TILE][TILE + 1];  // +1 消除 bank conflict

    // 1. 计算当前 thread 负责的全局元素坐标
    int gx = blockIdx.x * TILE + threadIdx.x;   // 全局列
    int gy = blockIdx.y * TILE + threadIdx.y;   // 全局行

    // 2. 边界保护
    if (gx < n && gy < n)
        tile[threadIdx.y][threadIdx.x] = in[gy * n + gx];

    __syncthreads();

    // 3. 转置后写回：行列互换
    int gx_t = blockIdx.y * TILE + threadIdx.x;  // 转置后的列
    int gy_t = blockIdx.x * TILE + threadIdx.y;  // 转置后的行

    if (gx_t < n && gy_t < n)
        out[gy_t * n + gx_t] = tile[threadIdx.x][threadIdx.y];
}

// -------------------- host --------------------
int main() {
    size_t bytes = N * N * sizeof(float);
    float *h_in = new float[N * N];
    float *h_out = new float[N * N];
    for (int i = 0; i < N * N; ++i) h_in[i] = i;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    transpose2D<<<grid, block>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    // 打印左上角 8×8 验证
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j)
            printf("%6.1f ", h_out[i * N + j]);
        printf("\n");
    }

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out;
    return 0;
}