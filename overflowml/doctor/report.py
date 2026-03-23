"""Doctor report — aggregates all checks into a DoctorReport."""

from __future__ import annotations

import platform
import sys
from typing import Optional

from ..core.types import DoctorReport
from .checks import (
    check_driver_mismatch,
    check_gpu,
    check_model_fit,
    check_optional_dep,
    check_python,
    check_ram,
    check_torch,
)

OPTIONAL_DEPS = [
    ("transformers", "transformers", "transformers"),
    ("accelerate", "accelerate", "transformers"),
    ("diffusers", "diffusers", "diffusers"),
    ("torchao", "torchao", "quantize"),
    ("bitsandbytes", "bitsandbytes", "bnb"),
]


def run(
    model: Optional[str] = None,
    model_size_gb: Optional[float] = None,
) -> DoctorReport:
    """Run all environment health checks and return a DoctorReport."""
    report = DoctorReport()

    # Environment info
    report.environment = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": f"{platform.system()}-{platform.machine()}",
    }
    try:
        import torch
        report.environment["torch"] = torch.__version__
        cuda_ver = getattr(torch.version, "cuda", None)
        hip_ver = getattr(torch.version, "hip", None)
        if cuda_ver:
            report.environment["torch_cuda"] = cuda_ver
        if hip_ver:
            report.environment["torch_hip"] = hip_ver
    except ImportError:
        report.environment["torch"] = "not installed"

    # Hardware info
    report.hardware = {}
    try:
        import torch
        if torch.cuda.is_available():
            report.hardware["gpu"] = torch.cuda.get_device_name(0)
            report.hardware["vram_gb"] = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.0f}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            report.hardware["gpu"] = "Apple Silicon (MPS)"
    except (ImportError, Exception):
        pass
    try:
        import psutil
        report.hardware["ram_gb"] = f"{psutil.virtual_memory().total / (1024**3):.0f}"
    except ImportError:
        pass

    # Run checks
    report.issues.append(check_python())
    report.issues.append(check_torch())
    report.issues.append(check_gpu())
    report.issues.append(check_ram())

    driver = check_driver_mismatch()
    if driver:
        report.issues.append(driver)

    for name, import_path, pip_extra in OPTIONAL_DEPS:
        report.issues.append(check_optional_dep(name, import_path, pip_extra))

    model_check = check_model_fit(model, model_size_gb)
    if model_check:
        report.issues.append(model_check)

    # Aggregate
    report.ok = all(i.severity != "error" for i in report.issues)
    report.fix_commands = [
        i.suggested_fix for i in report.issues
        if i.suggested_fix and i.severity in ("warn", "error")
    ]

    return report
