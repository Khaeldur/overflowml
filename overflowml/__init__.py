"""OverflowML — Run AI models larger than your GPU."""

__version__ = "0.5.0"

from .detect import detect_hardware, HardwareProfile
from .strategy import pick_strategy, Strategy, DistributionMode, MoEProfile, plan_llamacpp, get_moe_profile, MOE_REGISTRY
from .optimize import optimize_pipeline, optimize_model, MemoryGuard

def load_model(*args, **kwargs):
    """Load a HuggingFace model with optimal memory strategy. Lazy import."""
    from .transformers_ext import load_model as _load
    return _load(*args, **kwargs)

__all__ = [
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
