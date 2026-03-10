"""OverflowML — Run AI models larger than your GPU."""

__version__ = "0.1.0"

from .detect import detect_hardware, HardwareProfile
from .strategy import pick_strategy, Strategy
from .optimize import optimize_pipeline, optimize_model

__all__ = [
    "detect_hardware",
    "HardwareProfile",
    "pick_strategy",
    "Strategy",
    "optimize_pipeline",
    "optimize_model",
]
