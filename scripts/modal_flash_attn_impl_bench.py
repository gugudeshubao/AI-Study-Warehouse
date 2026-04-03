from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "flash-attn-impl-bench"
LOCAL_FLASH_ATTN_DIR = Path(__file__).resolve().parent.parent / "codes" / "flash-attn-handwritten"
REMOTE_FLASH_ATTN_DIR = "/root/flash-attn-handwritten"
SHAPES = [
    (4, 16, 16, 64),
    (4, 16, 32, 64),
    (4, 16, 64, 64),
]
ARCHES = [
    (1, "V2_CUDA"),
    (8, "V2_CUBLAS"),
]
ITERS = 200
WARMUP = 20

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


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
)
def build_and_benchmark() -> str:
    build_dir = os.path.join(REMOTE_FLASH_ATTN_DIR, "build_modal_impl_bench")
    logs: list[str] = []

    logs.append(run_command(["bash", "-lc", "nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader"], cwd=REMOTE_FLASH_ATTN_DIR))
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
    logs.append(run_command(["cmake", "--build", build_dir, "--target", "benchmark", "-j", "4"], cwd=REMOTE_FLASH_ATTN_DIR))

    benchmark_bin = os.path.join(build_dir, "benchmark")
    results: list[dict[str, object]] = []
    for B, H, N, D in SHAPES:
        for arch, _ in ARCHES:
            stdout = run_command(
                [
                    benchmark_bin,
                    str(B),
                    str(H),
                    str(N),
                    str(D),
                    str(arch),
                    str(ITERS),
                    str(WARMUP),
                ],
                cwd=REMOTE_FLASH_ATTN_DIR,
            )
            results.append(json.loads(stdout.strip().splitlines()[-1]))

    rendered = {
        "gpu": logs[0].strip(),
        "results": results,
    }
    print(json.dumps(rendered, ensure_ascii=True, indent=2), flush=True)
    return json.dumps(rendered, ensure_ascii=True, indent=2)


@app.local_entrypoint()
def main() -> None:
    print(build_and_benchmark.remote())
