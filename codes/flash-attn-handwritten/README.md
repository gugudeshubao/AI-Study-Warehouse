# flash-attn-handwritten

手写版 FlashAttention 实验仓，当前主线是 `V2` 前向/反向的多实现对比。

## 当前状态

- 已在 Modal `A10G (SM 8.6, CUDA 11.8)` 上实测通过：
  - `V2_CPU` 前向
  - `V2_CUDA` 前向 / 反向
  - `V2_CUBLAS` 前向 / 反向
  - `V2_AMPERE_WMMA` 前向
  - `V3_CUDA` 前向 / 反向
- 已接入但未实测通过：
  - `V2_AMPERE_MMA` 文件存在，但 dispatch 仍未接前向/反向
  - `Hopper / Blackwell / V3` 代码存在，但不在当前主构建链路

## 本地构建

```bash
cd codes/flash-attn-handwritten
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

如果 CMake 找不到 CUDA，可以显式指定：

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda
```

## 正确性测试

```bash
cd codes/flash-attn-handwritten
./build/test_correctness
./build/test_backward
```

当前 A10G 实测结果：

- `test_correctness`
  - `V2_CUBLAS` max diff: `1.013279e-06`
  - `V2_CUDA` max diff: `5.066395e-07`
  - `V2_AMPERE_WMMA` max diff: `4.768372e-07`
- `test_backward`
  - `V2_CUDA` max diff: `5.722046e-06`
  - `V2_CUBLAS` max diff: `5.960464e-06`
  - `V3_CUDA` max diff: `8.583069e-06`

## Benchmark

先构建：

```bash
cd codes/flash-attn-handwritten
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target benchmark -j
```

运行格式：

```bash
./build/benchmark B H N D arch [iters] [warmup]
```

当前可用的 `arch`：

- `0`: `V2_CPU`
- `1`: `V2_CUDA`
- `2`: `V2_AMPERE_WMMA`
- `8`: `V2_CUBLAS`
- `11`: `V3_CUDA`

示例：

```bash
./build/benchmark 2 16 2048 64 8 50 10
```

## Modal 脚本

仓库根目录下有两个可直接复用的脚本：

- `scripts/modal_flash_attn_ampere.py`
  - 在 `A10G` 上构建并跑 `test_correctness` / `test_backward`
- `scripts/modal_flash_attn_vs_tridao.py`
  - 在 `A10G` 上构建 `benchmark`
  - 对比当前实现和 Tri Dao 官方 `flash-attn`
- `scripts/modal_flash_attn_impl_bench.py`
  - 在 `A10G` 上对比当前仓库内部的 `V2_CUDA` / `V2_CUBLAS`
- `scripts/modal_flash_attn_v3_bwd_bench.py`
  - 在 `A10G` 上对比 `V2_CUBLAS backward` / `V3_CUDA backward`

运行：

```bash
modal run scripts/modal_flash_attn_ampere.py
modal run scripts/modal_flash_attn_vs_tridao.py
modal run scripts/modal_flash_attn_impl_bench.py
modal run scripts/modal_flash_attn_v3_bwd_bench.py
```

## 与 Tri Dao 官方实现的对比

测试环境：

- GPU: `A10G (SM 8.6)`
- 你的实现：`V3_CUDA`，`fp32`，forward-only
- 对照：Tri Dao `flash-attn==2.5.9.post1`，`fp16`，forward-only

实测结果：

| Shape | Our Impl | Our ms | Tri Dao ms | Tri Dao Faster |
|---|---|---:|---:|---:|
| `B4 H16 N512 D64` | `V3_CUDA` | `2.346` | `0.099` | `23.82x` |
| `B2 H16 N1024 D64` | `V3_CUDA` | `4.611` | `0.177` | `25.98x` |
| `B2 H16 N2048 D64` | `V3_CUDA` | `18.377` | `0.581` | `31.61x` |

对应吞吐：

- 你的实现：`1.83 ~ 1.87 TFLOP/s`
- Tri Dao：`43.53 ~ 59.09 TFLOP/s`

这个对比不是完全 apples-to-apples，因为：

- 你的路径当前是 `fp32`
- Tri Dao 官方实现跑的是 `fp16`
- 这里比较的是 forward-only，不含 backward

但趋势很明确：当前主瓶颈不是某一个 kernel 写得不够细，而是整体算法路径仍然是：

1. `QK^T`
2. 显式 materialize `S`
3. 独立 softmax
4. `PV`

而 Tri Dao 的核心优势是 fused tiled + online softmax，不需要把完整 `S` 落到全局内存。

## 仓库内部实现对比

在同一张 `A10G` 上，对小序列的前向 benchmark 做了额外对比：

| Shape | `V2_CUDA` ms | `V2_CUBLAS` ms | 更快实现 |
|---|---:|---:|---|
| `B4 H16 N16 D64` | `0.061` | `0.034` | `V2_CUBLAS` |
| `B4 H16 N32 D64` | `0.197` | `0.047` | `V2_CUBLAS` |
| `B4 H16 N64 D64` | `0.449` | `0.065` | `V2_CUBLAS` |

对应结论：

- 当前稳定路径里，`V2_CUBLAS` 从很小的序列长度开始就已经优于 `V2_CUDA`
- `V2_CUDA` 更像教学 / 参考实现，不是当前最佳性能路径
- 如果目标是继续追性能，应该优先改 `V2_CUBLAS` 这条主线的算法形态，而不是继续打磨当前的朴素 `V2_CUDA`

## V3 Streaming 原型

当前已经接入了一个真正的 `V3_CUDA` 前向原型：

- 不再显式 materialize 全量 `S`
- 用 tiled + online softmax 做前向累加
- A10G 上 correctness 通过
  - `V3_CUDA` fwd max diff: `1.013279e-06`
  - `V3_CUDA` bwd max diff: `1.144409e-05`

和 `V2_CUBLAS` 的 A10G 实测对比如下：

| Shape | `V2_CUBLAS` ms | `V3_CUDA` ms | 更快实现 |
|---|---:|---:|---|
| `B4 H16 N64 D64` | `0.066` | `0.055` | `V3_CUDA` |
| `B4 H16 N128 D64` | `0.328` | `0.177` | `V3_CUDA` |
| `B2 H16 N256 D64` | `0.565` | `0.341` | `V3_CUDA` |
| `B2 H16 N512 D64` | `1.927` | `1.222` | `V3_CUDA` |

对应解读：

- `V3_CUDA` 方向是对的：它已经摆脱了 `N^2` 共享内存/中间矩阵落地的结构
- 经过 `warp` 内双 `subwarp` 并行、subwarp 协作 softmax、shared `K/V` tile 复用和寄存器化 `Q/acc` 后，当前测试区间内已经全面超过 `V2_CUBLAS`
- 当前 `V3_CUDA` backward 已经接成了真正的 tiled CUDA kernel，并在 A10G 上通过了 correctness
- 但它还不是高性能版 backward：
  - 当前目标是先闭合 `V3` 前后向链路
  - 性能优化仍然主要集中在 backward 主线

当前 `V3_CUDA backward` 和 `V2_CUBLAS backward` 的 A10G benchmark：

| Shape | `V2_CUBLAS bwd` ms | `V3_CUDA bwd` ms | 更快实现 |
|---|---:|---:|---|
| `B2 H16 N128 D64` | `14.134` | `1.125` | `V3_CUDA` |
| `B2 H16 N256 D64` | `26.198` | `3.856` | `V3_CUDA` |
| `B2 H16 N512 D64` | `67.550` | `13.588` | `V3_CUDA` |

对应解读：

- `V3_CUDA backward` 现在已经在当前测试区间内全面超过 `V2_CUBLAS backward`
- 这次收益主要来自：
  - 多行 block 并行
  - tile 内 `dK / dV` 先在 shared memory 聚合，再一次性写回全局
  - softmax backward 的整行统计与 tile 局部统计拆开
- 和 Tri Dao backward 的最新 A10G 对比：

| Shape | `V3_CUDA bwd` ms | `Tri Dao bwd` ms | Tri Dao Faster |
|---|---:|---:|---:|
| `B2 H16 N128 D64` | `1.129` | `0.174` | `6.50x` |
| `B2 H16 N256 D64` | `3.803` | `0.170` | `22.36x` |
| `B2 H16 N512 D64` | `12.670` | `0.178` | `71.21x` |

- 如果继续优化，重点会是：
  - 减少 backward 里的重复 QK / doutV 重算
  - 进一步压缩全局写回次数
  - 把 backward 的逐行三遍扫描继续合并
- 如果继续推进 `V3`，下一步重点不再是“correctness”，而是：
  - 更粗粒度的 tile 并行
  - 更好的 Q/K/V shared memory 复用
  - warp-level reduction / vectorized loads
  - 尽量向 Tensor Core 友好的数据路径靠拢

## 现阶段结论

- 正确性主线已经稳定：
  - `CPU / CUDA / CUBLAS` 的 `V2` 前后向可用
- 性能主线还远不是官方 FlashAttention：
  - 当前最快、最稳定的前向路径已经切到 `V3_CUDA`
  - `V3_CUDA` 已经是 streaming / online softmax 路线
  - `V3_CUDA` backward 已可用，而且已经开始做性能优化
  - 与官方 `flash-attn` 的差距已经从之前的 `40x ~ 100x` 缩到当前的 `24x ~ 32x`

## 下一步

- 把 `V2_AMPERE_WMMA` backward 接入并实测
- 决定是否继续维护 `V2_AMPERE_MMA`
- 如果目标是接近 Tri Dao，需要改算法而不只是换 Tensor Core 指令：
  - tiled streaming
  - online softmax
  - 避免 materialize 全量 `S`
