#!/usr/bin/env python3
"""
Thor U NVFP4 (e2m1) 数值验证 + 性能 benchmark
使用 PyTorch _scaled_mm 走 cuBLASLt NVFP4 路径

用法:
    # Thor U 上（需要 wallx-venv 或任何有 torch 2.11+ 的环境）
    source /home/user/wy/wallx-venv/bin/activate
    python3 thoru_nvfp4_bench.py

编码参考:
    - NVFP4 e2m1 格式: sign(1) + exp(2) + mantissa(1), 4-bit
    - e2m1 1.0 = 0x2 (二进制 0010), packed pair = 0x22
    - Scale e4m3 1.0 = 0x38
    - PyTorch dtype: torch.float4_e2m1fn_x2 (每 byte 存 2 个 e2m1)
    - _scaled_mm blockwise 1x16 scaling: 每 16 个 e2m1 元素共享 1 个 e4m3 scale
"""

import sys
import torch


def check_environment():
    """检查运行环境"""
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU:     {gpu_name}")
    print(f"SM:      {torch.cuda.get_device_capability(0)}")

    has_fp4 = hasattr(torch, "float4_e2m1fn_x2")
    print(f"FP4 dtype (float4_e2m1fn_x2): {'available' if has_fp4 else 'NOT available'}")
    if not has_fp4:
        print("ERROR: torch.float4_e2m1fn_x2 not supported in this PyTorch version")
        sys.exit(1)

    return gpu_name


def create_nvfp4_tensors(M, N, K, device, fill_val=0x22, scale_val=0x38):
    """
    创建 NVFP4 测试张量

    Args:
        fill_val: e2m1 packed byte, 0x22 = 两个 1.0
        scale_val: e4m3 scale byte, 0x38 = 1.0
    """
    packed_K = K // 2  # float4_e2m1fn_x2 每 byte 存 2 个值
    a_bytes = torch.full((M, packed_K), fill_val, dtype=torch.uint8, device=device)
    b_bytes = torch.full((N, packed_K), fill_val, dtype=torch.uint8, device=device)
    a_fp4 = a_bytes.view(torch.float4_e2m1fn_x2)
    b_fp4 = b_bytes.view(torch.float4_e2m1fn_x2)

    # blockwise 1x16 scaling: 每 16 个 e2m1 元素 1 个 scale
    scale_block_size = 16
    num_scale_a = M * K // scale_block_size
    num_scale_b = N * K // scale_block_size
    sa_bytes = torch.full((num_scale_a,), scale_val, dtype=torch.uint8, device=device)
    sb_bytes = torch.full((num_scale_b,), scale_val, dtype=torch.uint8, device=device)
    scale_a = sa_bytes.view(torch.float8_e4m3fn)
    scale_b = sb_bytes.view(torch.float8_e4m3fn)

    return a_fp4, b_fp4, scale_a, scale_b


def numerical_validation(device):
    """全面的数值正确性验证"""
    print("\n" + "=" * 70)
    print("NVFP4 数值正确性验证")
    print("=" * 70)
    print("输入: A=B=全 e2m1 1.0 (0x22), scale=全 e4m3 1.0 (0x38)")
    print("预期: D[i][j] = K (每行 K 个 1.0*1.0 累加)")
    print("-" * 70)
    print(f"{'M':>6} {'N':>6} {'K':>6} | {'out[0,0]':>10} {'expected':>10} {'match':>6} | {'all_correct':>12}")
    print("-" * 70)

    test_sizes = [
        (128, 128, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]

    total_pass = 0
    total_tests = 0

    for M, N, K in test_sizes:
        try:
            a_fp4, b_fp4, scale_a, scale_b = create_nvfp4_tensors(M, N, K, device)
            out = torch._scaled_mm(
                a_fp4, b_fp4.t(),
                scale_a=scale_a, scale_b=scale_b,
                out_dtype=torch.float32,
            )
            actual = out[0, 0].item()
            expected = float(K)
            match = actual == expected
            ref = torch.full((M, N), expected, device=device)
            all_correct = torch.allclose(out, ref)
            sym = "YES" if match else "NO"
            all_sym = "ALL CORRECT" if all_correct else "MISMATCH"
            print(f"{M:>6} {N:>6} {K:>6} | {actual:>10.1f} {expected:>10.1f} {sym:>6} | {all_sym:>12}")
            if all_correct:
                total_pass += 1
            total_tests += 1
        except Exception as exc:
            msg = str(exc)[:60]
            print(f"{M:>6} {N:>6} {K:>6} | ERROR: {msg}")
            total_tests += 1

    print("-" * 70)
    print(f"Result: {total_pass}/{total_tests} passed")
    return total_pass == total_tests


def performance_benchmark(device, warmup=5, iters=20):
    """NVFP4 GEMM 性能 benchmark"""
    print("\n" + "=" * 70)
    print("NVFP4 GEMM 性能 Benchmark")
    print("=" * 70)
    print(f"{'M':>6} {'N':>6} {'K':>6} | {'avg_ms':>9} {'min_ms':>9} | {'TFLOPS':>8}")
    print("-" * 70)

    bench_sizes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]

    for M, N, K in bench_sizes:
        try:
            a_fp4, b_fp4, scale_a, scale_b = create_nvfp4_tensors(M, N, K, device)

            # warmup
            for _ in range(warmup):
                torch._scaled_mm(a_fp4, b_fp4.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float32)
            torch.cuda.synchronize()

            # benchmark
            timings = []
            for _ in range(iters):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                torch._scaled_mm(a_fp4, b_fp4.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float32)
                end_event.record()
                torch.cuda.synchronize()
                timings.append(start_event.elapsed_time(end_event))

            avg_ms = sum(timings) / len(timings)
            min_ms = min(timings)
            tflops = 2 * M * N * K / min_ms / 1e9
            print(f"{M:>6} {N:>6} {K:>6} | {avg_ms:>8.3f}ms {min_ms:>8.3f}ms | {tflops:>7.1f}")
        except Exception as exc:
            msg = str(exc)[:50]
            print(f"{M:>6} {N:>6} {K:>6} | ERROR: {msg}")

    print("=" * 70)


def main():
    gpu_name = check_environment()
    device = torch.device("cuda:0")

    all_pass = numerical_validation(device)
    performance_benchmark(device)

    print(f"\n{'=' * 70}")
    if all_pass:
        print(f"CONCLUSION: {gpu_name} NVFP4 cuBLASLt ALL NUMERICAL TESTS PASSED")
    else:
        print(f"CONCLUSION: {gpu_name} NVFP4 cuBLASLt SOME TESTS FAILED")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
