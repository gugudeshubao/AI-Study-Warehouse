from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "flash-attn-handwritten-ampere-test"
LOCAL_FLASH_ATTN_DIR = Path(__file__).resolve().parent.parent / "codes" / "flash-attn-handwritten"
REMOTE_FLASH_ATTN_DIR = "/root/flash-attn-handwritten"

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
    return output


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
)
def build_and_test() -> str:
    build_dir = os.path.join(REMOTE_FLASH_ATTN_DIR, "build_modal")
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
    logs.append(run_command(["cmake", "--build", build_dir, "-j", "4"], cwd=REMOTE_FLASH_ATTN_DIR))
    logs.append(run_command([os.path.join(build_dir, "test_correctness")], cwd=REMOTE_FLASH_ATTN_DIR))
    logs.append(run_command([os.path.join(build_dir, "test_backward")], cwd=REMOTE_FLASH_ATTN_DIR))
    return "\n\n".join(logs)


@app.local_entrypoint()
def main() -> None:
    print(build_and_test.remote())
