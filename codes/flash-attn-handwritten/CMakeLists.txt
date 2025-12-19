cmake_minimum_required(VERSION 3.20)
project(flash_attn_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 70)   # Ampere 起步，够用

# ---- 找 CUDA / cuBLAS（后续用） ----
find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)       # 系统已装 libgtest-dev 或源码
# 我电脑的cublas的路径有点特殊，所以我手动设置一下，默认路径不用设置
set(CUBLAS_TEMP_ROOT_PATH /usr/local/cuda-11.4/targets/x86_64-linux/include)

add_library(flash_attn STATIC
    src/flash_attn_dispatch.cpp
    src/cpu/flash_attn_v2_cpu.cpp
    src/cuda/flash_attn_v2_cuda.cu
    src/baseline/flash_attn_v2_baseline.cu   # <-- 新增
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

# # ---- 可执行：性能测试 ----
# add_executable(benchmark tests/benchmark.cpp)
# target_link_libraries(benchmark PRIVATE flash_attn)

# target_include_directories(benchmark
#     PRIVATE
#         ${CMAKE_SOURCE_DIR}/include
#         ${CMAKE_SOURCE_DIR}/src
#         ${CMAKE_SOURCE_DIR}/src
#         ${CUBLAS_TEMP_ROOT_PATH}
# )