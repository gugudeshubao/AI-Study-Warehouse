cmake_minimum_required(VERSION 3.20)
project(flash_attn_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)   # Ampere 起步，够用

# ---- 找 CUDA / cuBLAS（后续用） ----
find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)       # 系统已装 libgtest-dev 或源码
# 我电脑的cublas的路径有点特殊，所以我手动设置一下，默认路径不用设置
set(CUBLAS_TEMP_ROOT_PATH /usr/local/cuda-11.4/targets/x86_64-linux/include)

add_library(flash_attn STATIC
    src/flash_attn_dispatch.cpp
    src/cpu/flash_attn_v2_cpu.cpp
    src/cpu/flash_attn_v3_cpu.cpp
    src/cuda/flash_attn_v2_cuda.cu
    src/cuda/flash_attn_v3_cuda.cu
    src/baseline/flash_attn_v2_baseline.cu
    src/ampere/flash_attn_v2_ampere_wmma.cu
    src/ampere/flash_attn_v2_ampere_mma.cu
    # src/hopper/flash_attn_v2_hopper_wgmma.cu
    # src/blackwell/flash_attn_v2_blackwell_tcgen05.cu
)
target_include_directories(flash_attn
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src
        ${CUBLAS_TEMP_ROOT_PATH}
)
target_link_libraries(flash_attn PUBLIC CUDA::cudart CUDA::cublas)


add_executable(test_correctness tests/test_correctness.cpp)
target_link_libraries(test_correctness PRIVATE flash_attn GTest::gtest GTest::gtest_main)
target_include_directories(test_correctness
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CUBLAS_TEMP_ROOT_PATH}
)

add_executable(test_backward tests/test_backward.cpp)
target_link_libraries(test_backward PRIVATE flash_attn GTest::gtest GTest::gtest_main)
target_include_directories(test_backward
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CUBLAS_TEMP_ROOT_PATH}
)

# ---- 可执行：性能测试 ----
add_executable(benchmark tests/benchmark.cpp)
target_link_libraries(benchmark PRIVATE flash_attn)

target_include_directories(benchmark
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src
        ${CUBLAS_TEMP_ROOT_PATH}
)

add_executable(benchmark_backward tests/benchmark_backward.cpp)
target_link_libraries(benchmark_backward PRIVATE flash_attn)

target_include_directories(benchmark_backward
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CUBLAS_TEMP_ROOT_PATH}
)


# 调试程序
add_executable(test_block_index tests/test_block_index.cu)
target_link_libraries(test_block_index PRIVATE CUDA::cudart)

target_include_directories(test_block_index
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src
        ${CUBLAS_TEMP_ROOT_PATH}
)
