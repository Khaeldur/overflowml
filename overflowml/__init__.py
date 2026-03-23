"""OverflowML — Run AI models larger than your GPU."""

__version__ = "0.12.0"

# --- New public API (preferred) ---
from . import doctor
from .inspect import inspect_model
from .core.planner import plan
from .core.can_run import can_run
from .monitor import Monitor, MetricsExporter
from .batch import auto_batch, calculate_batch_size, BatchConfig, measure_vram_headroom
from .core.types import (
    ModelInfo,
    HardwareInfo,
    StrategyCandidate,
    PlanResult,
    DoctorIssue,
    DoctorReport,
    CanRunResult,
    GPUInfo,
)

# --- Legacy API (still supported) ---
import warnings as _w
with _w.catch_warnings():
    _w.simplefilter("ignore", DeprecationWarning)
    from .detect import detect_hardware, HardwareProfile
    from .strategy import pick_strategy, Strategy, DistributionMode, MoEProfile, plan_llamacpp, get_moe_profile, MOE_REGISTRY
    from .optimize import optimize_pipeline, optimize_model, MemoryGuard


def load_model(*args, **kwargs):
    """Load a HuggingFace model with optimal memory strategy. Lazy import."""
    from .transformers_ext import load_model as _load
    return _load(*args, **kwargs)


__all__ = [
    # New API
    "doctor",
    "inspect_model",
    "plan",
    "can_run",
    "Monitor",
    "MetricsExporter",
    "auto_batch",
    "calculate_batch_size",
    "BatchConfig",
    "measure_vram_headroom",
    "ModelInfo",
    "HardwareInfo",
    "StrategyCandidate",
    "PlanResult",
    "DoctorIssue",
    "DoctorReport",
    "CanRunResult",
    "GPUInfo",
    # Legacy API
    "detect_hardware",
    "HardwareProfile",
    "pick_strategy",
    "Strategy",
    "MoEProfile",
    "MOE_REGISTRY",
    "get_moe_profile",
    "plan_llamacpp",
    "optimize_pipeline",
    "optimize_model",
    "load_model",
    "MemoryGuard",
]
