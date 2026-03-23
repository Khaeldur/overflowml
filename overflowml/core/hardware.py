"""Hardware detection — wraps legacy detect.py and returns new HardwareInfo."""

from __future__ import annotations

import platform
import sys
from typing import Optional

from .types import GPUInfo, HardwareInfo


def detect_hardware_info() -> HardwareInfo:
    """Detect hardware and return a HardwareInfo object."""
    hw = HardwareInfo(platform=f"{platform.system()}-{platform.machine()}")

    try:
        import psutil
        hw.total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        hw.cpu_cores = psutil.cpu_count(logical=False) or 1
    except ImportError:
        pass

    _detect_torch(hw)
    return hw


def _detect_torch(hw: HardwareInfo) -> None:
    """Fill in torch-related fields."""
    try:
        import torch
        hw.torch_version = torch.__version__
    except ImportError:
        return

    import torch

    # CUDA / ROCm
    if torch.cuda.is_available():
        is_rocm = getattr(torch.version, "hip", None) is not None
        hw.torch_cuda_available = True
        hw.torch_cuda_version = getattr(torch.version, "cuda", None)
        hw.torch_hip_version = getattr(torch.version, "hip", None)
        backend = "rocm" if is_rocm else "cuda"

        compute_caps = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram = props.total_memory / (1024 ** 3)
            compute_caps.append((props.major, props.minor))
            hw.gpus.append(GPUInfo(
                name=props.name,
                total_vram_gb=vram,
                backend=backend,
                device_index=i,
            ))

        hw.supports_bf16 = all(cc >= (8, 0) for cc in compute_caps)

        if not is_rocm:
            try:
                from torchao.quantization import Float8WeightOnlyConfig  # noqa: F401
                hw.supports_fp8 = True
            except ImportError:
                pass
        return

    # MPS
    if sys.platform == "darwin":
        try:
            if torch.backends.mps.is_available():
                hw.unified_memory = True
                hw.supports_bf16 = True
                name = "Apple Silicon"
                try:
                    import subprocess
                    result = subprocess.run(
                        ["sysctl", "-n", "machdep.cpu.brand_string"],
                        capture_output=True, text=True,
                    )
                    if result.returncode == 0:
                        name = result.stdout.strip()
                except Exception:
                    pass
                hw.gpus.append(GPUInfo(
                    name=name,
                    total_vram_gb=hw.total_ram_gb,
                    backend="mps",
                    device_index=0,
                ))
                return
        except Exception:
            pass

        # MLX
        try:
            import mlx.core  # noqa: F401
            hw.unified_memory = True
            hw.supports_bf16 = True
            hw.gpus.append(GPUInfo(
                name="Apple Silicon (MLX)",
                total_vram_gb=hw.total_ram_gb,
                backend="mps",
                device_index=0,
            ))
            return
        except ImportError:
            pass


def hardware_info_to_legacy(hw_info: HardwareInfo):
    """Convert new HardwareInfo to legacy HardwareProfile for backward compat."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ..detect import Accelerator, HardwareProfile

    backend_map = {
        "cuda": Accelerator.CUDA,
        "rocm": Accelerator.ROCm,
        "mps": Accelerator.MPS,
        "cpu": Accelerator.CPU,
    }
    accel = backend_map.get(hw_info.primary_backend, Accelerator.CPU)

    gpu_vram_gbs = [g.total_vram_gb for g in hw_info.gpus]
    min_vram = min(gpu_vram_gbs) if gpu_vram_gbs else 0.0

    return HardwareProfile(
        accelerator=accel,
        gpu_name=hw_info.gpus[0].name if hw_info.gpus else "None",
        gpu_vram_gb=min_vram,
        system_ram_gb=hw_info.total_ram_gb,
        unified_memory=hw_info.unified_memory,
        os=hw_info.platform.split("-")[0] if hw_info.platform else "",
        cpu_cores=hw_info.cpu_cores,
        supports_bf16=hw_info.supports_bf16,
        supports_fp8=hw_info.supports_fp8,
        num_gpus=hw_info.num_gpus,
        gpu_names=[g.name for g in hw_info.gpus],
        gpu_vram_gbs=gpu_vram_gbs,
        total_gpu_vram_gb=hw_info.total_vram_gb,
    )
