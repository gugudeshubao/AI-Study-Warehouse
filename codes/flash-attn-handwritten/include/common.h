#pragma once
#include <cstddef>

#define CHECK_CUDA_LAST() do {                                          \
    cudaError_t err = cudaGetLastError();                               \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "[%s:%d] CUDA launch error: errnum:%d,errmsg:%s\n",              \
                __FILE__, __LINE__, err,cudaGetErrorString(err));           \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while (0)

#define CHECK_CUDA(ans) do {                                           \
    cudaError_t err_ = (ans);                                          \
    if (err_ != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error at %s:%d : errnum:%d,errmsg:%s\n",                  \
                __FILE__, __LINE__, err_,cudaGetErrorString(err_));         \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while (0)