
# test case
```

mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
./test_correctness

```
```
如果报错“Couldn't find CUDA library root.”，可以手动设置CMAKE_CUDA_COMPIL，如下所示：
cmake ..  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.4/bin/nvcc
```

# benchemark test
```
cd build
make benchmark
./benchmark 2 16 2048 64 2     # arch=2 即 V2_CUBLAS
```