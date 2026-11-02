"""GPU setup shared by the baseline and the engine.

The memory cap keeps PyTorch from taking the whole 6 GB card, so the desktop,
browser and VS Code stay responsive while a server is running. It caps the
PyTorch caching allocator only; the CUDA context adds roughly 300-500 MB on top.

Override the cap with the MINI_INFER_GPU_FRACTION environment variable
(e.g. 0.6) or the --gpu-fraction flag on the server.
"""

import os
import subprocess

import torch

DEFAULT_GPU_FRACTION = 0.75


def setup_device(gpu_fraction: float | None = None) -> torch.device:
    if not torch.cuda.is_available():
        print("[device] CUDA not available, falling back to CPU")
        return torch.device("cpu")

    if gpu_fraction is None:
        gpu_fraction = float(os.environ.get("MINI_INFER_GPU_FRACTION", DEFAULT_GPU_FRACTION))
    if not 0.0 < gpu_fraction <= 1.0:
        raise ValueError(f"gpu_fraction must be in (0, 1], got {gpu_fraction}")

    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(gpu_fraction, device)
    total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(
        f"[device] {torch.cuda.get_device_name(device)}: "
        f"capped at {gpu_fraction:.0%} of {total_gb:.1f} GB ({gpu_fraction * total_gb:.1f} GB)"
    )
    return device


def gpu_info() -> dict:
    """Hardware/software details recorded in every benchmark result."""
    info = {"torch": torch.__version__, "cuda": torch.version.cuda}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(props.total_memory / 1e9, 2)
        try:
            info["driver"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            info["driver"] = None
    return info


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None
