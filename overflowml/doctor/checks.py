"""Individual environment health checks."""

from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from typing import Optional

from ..core.types import DoctorIssue


def check_python() -> DoctorIssue:
    v = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info < (3, 10):
        return DoctorIssue(
            code="python_old", severity="error",
            message=f"Python {v} — OverflowML requires 3.10+",
            detected_value=v,
            suggested_fix="Install Python 3.10 or newer",
        )
    return DoctorIssue(
        code="python_ok", severity="info",
        message=f"Python {v}",
        detected_value=v,
    )


def check_torch() -> DoctorIssue:
    try:
        import torch
        version = torch.__version__

        if torch.cuda.is_available():
            cuda_ver = getattr(torch.version, "cuda", "unknown")
            hip_ver = getattr(torch.version, "hip", None)
            if hip_ver:
                return DoctorIssue(
                    code="torch_rocm", severity="info",
                    message=f"PyTorch {version} (ROCm {hip_ver})",
                    detected_value=version,
                )
            return DoctorIssue(
                code="torch_cuda", severity="info",
                message=f"PyTorch {version} (CUDA {cuda_ver})",
                detected_value=version,
            )

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return DoctorIssue(
                code="torch_mps", severity="info",
                message=f"PyTorch {version} (MPS/Apple Silicon)",
                detected_value=version,
            )

        # CPU-only build
        return DoctorIssue(
            code="torch_cpu_only", severity="warn",
            message=f"PyTorch {version} (CPU-only build)",
            detected_value=version,
            suggested_fix="pip install torch --index-url https://download.pytorch.org/whl/cu124",
        )
    except ImportError:
        return DoctorIssue(
            code="torch_missing", severity="error",
            message="PyTorch not installed",
            suggested_fix="pip install torch",
        )


def check_gpu() -> DoctorIssue:
    gpu_name = _detect_system_gpu()

    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return DoctorIssue(
                code="gpu_ok", severity="info",
                message=f"GPU: {name} ({vram:.0f}GB VRAM)",
                detected_value=name,
            )
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return DoctorIssue(
                code="gpu_mps", severity="info",
                message="GPU: Apple Silicon (unified memory)",
            )
    except (ImportError, Exception):
        pass

    if gpu_name:
        return DoctorIssue(
            code="gpu_not_accessible", severity="error",
            message=f"GPU detected by system ({gpu_name}) but not accessible by PyTorch",
            detected_value=gpu_name,
            suggested_fix="Install CUDA-enabled PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu124",
        )

    return DoctorIssue(
        code="gpu_none", severity="warn",
        message="No GPU detected — CPU only",
        suggested_fix="Models will run on CPU (very slow)",
    )


def check_driver_mismatch() -> Optional[DoctorIssue]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        torch_cuda = getattr(torch.version, "cuda", None)
        if not torch_cuda:
            return None
    except ImportError:
        return None

    driver_cuda = _get_nvidia_smi_cuda_version()
    if not driver_cuda:
        return None

    # Compare major.minor
    torch_parts = torch_cuda.split(".")[:2]
    driver_parts = driver_cuda.split(".")[:2]
    if torch_parts[0] != driver_parts[0]:
        return DoctorIssue(
            code="driver_mismatch", severity="warn",
            message=f"Torch CUDA {torch_cuda} vs driver CUDA {driver_cuda} — major version mismatch",
            detected_value=f"torch={torch_cuda}, driver={driver_cuda}",
            suggested_fix=f"pip install torch --index-url https://download.pytorch.org/whl/cu{driver_parts[0]}{driver_parts[1]}",
        )
    return None


def check_ram() -> DoctorIssue:
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        if ram_gb < 16:
            return DoctorIssue(
                code="ram_low", severity="warn",
                message=f"System RAM: {ram_gb:.0f}GB — low for CPU offloading",
                detected_value=f"{ram_gb:.0f}GB",
            )
        return DoctorIssue(
            code="ram_ok", severity="info",
            message=f"System RAM: {ram_gb:.0f}GB",
            detected_value=f"{ram_gb:.0f}GB",
        )
    except ImportError:
        return DoctorIssue(
            code="psutil_missing", severity="warn",
            message="psutil not installed — cannot check RAM",
            suggested_fix="pip install psutil",
        )


def check_optional_dep(name: str, import_path: str, pip_extra: str) -> DoctorIssue:
    try:
        mod = importlib.import_module(import_path)
        version = getattr(mod, "__version__", "installed")
        return DoctorIssue(
            code=f"dep_{name}_ok", severity="info",
            message=f"{name} {version}",
            detected_value=version,
        )
    except ImportError:
        return DoctorIssue(
            code=f"dep_{name}_missing", severity="warn",
            message=f"{name} not installed",
            suggested_fix=f"pip install overflowml[{pip_extra}]",
        )


def check_model_fit(model: Optional[str], model_size_gb: Optional[float]) -> Optional[DoctorIssue]:
    if not model and not model_size_gb:
        return None

    size = model_size_gb
    model_label = f"{model_size_gb:.0f}GB model" if model_size_gb else model

    if model and not size:
        try:
            from ..inspect import estimate_size_gb
            size = estimate_size_gb(model)
            model_label = f"{model} (~{size:.0f}GB fp16)"
        except Exception:
            return DoctorIssue(
                code="model_unknown", severity="warn",
                message=f"Could not estimate size for {model}",
            )

    if not size:
        return None

    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if size * 1.15 <= vram:
                return DoctorIssue(
                    code="model_fits", severity="info",
                    message=f"{model_label} fits in VRAM ({vram:.0f}GB)",
                )
            else:
                return DoctorIssue(
                    code="model_needs_offload", severity="warn",
                    message=f"{model_label} exceeds VRAM ({vram:.0f}GB) — will need offloading or quantization",
                    suggested_fix=f"overflowml plan {model or size} --compare",
                )
    except (ImportError, Exception):
        pass

    return DoctorIssue(
        code="model_no_gpu", severity="warn",
        message=f"{model_label} — no GPU available for fit check",
    )


# ---- Helpers ----

def _detect_system_gpu() -> Optional[str]:
    """Try to detect GPU via nvidia-smi even if torch can't see it."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _get_nvidia_smi_cuda_version() -> Optional[str]:
    """Get CUDA version from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "CUDA Version" in line:
                    parts = line.split("CUDA Version:")
                    if len(parts) > 1:
                        return parts[1].strip().split()[0].rstrip("|").strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None
