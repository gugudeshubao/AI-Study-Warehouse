from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "flash-attn-v3-bwd-vs-tridao"
LOCAL_FLASH_ATTN_DIR = Path(__file__).resolve().parent.parent / "codes" / "flash-attn-handwritten"
REMOTE_FLASH_ATTN_DIR = "/root/flash-attn-handwritten"
BENCH_SHAPES = [
    (2, 16, 128, 64),
    (2, 16, 256, 64),
    (2, 16, 512, 64),
]
OUR_ARCH = 11  # Arch::V3_CUDA
ITERS = 20
WARMUP = 5

image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "build-essential",
        "cmake",
        "git",
        "libgtest-dev",
        "ninja-build",
    )
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel packaging",
        "python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.2.2",
        "TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4 python -m pip install flash-attn==2.5.9.post1 --no-build-isolation",
    )
    .add_local_dir(
        str(LOCAL_FLASH_ATTN_DIR),
        remote_path=REMOTE_FLASH_ATTN_DIR,
    )
)

app = modal.App(APP_NAME)


def run_command(cmd: list[str], cwd: str) -> str:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
    )
    rendered = [
        f"$ {' '.join(cmd)}",
        f"exit_code={proc.returncode}",
    ]
    if proc.stdout:
        rendered.append("stdout:")
        rendered.append(proc.stdout)
    if proc.stderr:
        rendered.append("stderr:")
        rendered.append(proc.stderr)
    output = "\n".join(rendered)
    print(output, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(output)
    return proc.stdout


def benchmark_tridao_backward(B: int, H: int, N: int, D: int, iters: int, warmup: int) -> dict[str, float | int | str]:
    import torch

    try:
        from flash_attn import flash_attn_func
    except ImportError:
        from flash_attn.flash_attn_interface import flash_attn_func

    q = torch.randn(B, N, H, D, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(B, N, H, D, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(B, N, H, D, device="cuda", dtype=torch.float16, requires_grad=True)
    dout = torch.randn(B, N, H, D, device="cuda", dtype=torch.float16)

    out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
    for _ in range(warmup):
        torch.autograd.grad(out, (q, k, v), dout, retain_graph=True)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        torch.autograd.grad(out, (q, k, v), dout, retain_graph=True)
    stop.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(stop) / iters
    flops = 8.0 * B * H * N * N * D
    tflops = flops / ms / 1e9
    return {
        "b": B,
        "h": H,
        "n": N,
        "d": D,
        "impl": "TriDao_flash_attn_bwd",
        "dtype": "fp16",
        "iters": iters,
        "warmup": warmup,
        "ms": ms,
        "tflops": tflops,
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
)
def build_and_benchmark() -> str:
    build_dir = os.path.join(REMOTE_FLASH_ATTN_DIR, "build_modal_bwd_vs_tridao")

    logs: list[str] = []
    logs.append(run_command(["bash", "-lc", "nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader"], cwd=REMOTE_FLASH_ATTN_DIR))
    logs.append(run_command(["bash", "-lc", "nvcc --version"], cwd=REMOTE_FLASH_ATTN_DIR))
    logs.append(
        run_command(
            [
                "cmake",
                "-S",
                ".",
                "-B",
                build_dir,
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
                "-DCUDAToolkit_ROOT=/usr/local/cuda",
            ],
            cwd=REMOTE_FLASH_ATTN_DIR,
        )
    )
    logs.append(run_command(["cmake", "--build", build_dir, "--target", "benchmark_backward", "-j", "4"], cwd=REMOTE_FLASH_ATTN_DIR))

    benchmark_bin = os.path.join(build_dir, "benchmark_backward")
    comparisons: list[dict[str, float | int | str]] = []
    for B, H, N, D in BENCH_SHAPES:
        ours_stdout = run_command(
            [
                benchmark_bin,
                str(B),
                str(H),
                str(N),
                str(D),
                str(OUR_ARCH),
                str(ITERS),
                str(WARMUP),
            ],
            cwd=REMOTE_FLASH_ATTN_DIR,
        )
        ours = json.loads(ours_stdout.strip().splitlines()[-1])
        tridao = benchmark_tridao_backward(B, H, N, D, ITERS, WARMUP)
        comparisons.append(
            {
                "shape": f"B{B}_H{H}_N{N}_D{D}",
                "ours_impl": ours["arch_name"],
                "ours_dtype": "fp32",
                "ours_ms": ours["ms"],
                "ours_tflops": ours["tflops"],
                "tridao_dtype": tridao["dtype"],
                "tridao_ms": round(tridao["ms"], 3),
                "tridao_tflops": round(tridao["tflops"], 2),
                "tridao_faster_x": round(float(ours["ms"]) / float(tridao["ms"]), 2),
            }
        )

    rendered = {
        "gpu": logs[0].strip(),
        "note": "Our benchmark uses fp32 backward and compares against Tri Dao flash-attn fp16 backward, so the ratio is directional rather than apples-to-apples.",
        "comparisons": comparisons,
    }
    print(json.dumps(rendered, ensure_ascii=True, indent=2), flush=True)
    return json.dumps(rendered, ensure_ascii=True, indent=2)


@app.local_entrypoint()
def main() -> None:
    print(build_and_benchmark.remote())
