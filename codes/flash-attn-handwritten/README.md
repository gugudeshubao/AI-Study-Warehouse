
# test case
```

mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
./test_correctness
```
# benchemark test
```
cd build
make benchmark
./benchmark 2 16 2048 64 2     # arch=2 即 V2_CUBLAS
```