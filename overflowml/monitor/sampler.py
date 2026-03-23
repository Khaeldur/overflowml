"""Memory and performance sampler for live monitoring."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Sample:
    timestamp: float = 0.0
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    gpu_name: str = ""


class Monitor:
    """Attach to a model or pipeline to monitor memory usage.

    Usage:
        from overflowml.monitor import Monitor
        mon = Monitor()
        mon.start()  # starts background sampling (or use as context manager)
    """

    def __init__(self, interval: float = 1.0, threshold: float = 0.85):
        self.interval = interval
        self.threshold = threshold
        self.samples: list[Sample] = []
        self._running = False

    def sample(self) -> Sample:
        """Take a single memory sample."""
        s = Sample(timestamp=time.time())
        try:
            import psutil
            mem = psutil.virtual_memory()
            s.ram_used_gb = mem.used / (1024 ** 3)
            s.ram_total_gb = mem.total / (1024 ** 3)
        except ImportError:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                s.vram_used_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                s.vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                s.gpu_name = torch.cuda.get_device_name(0)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                s.gpu_name = "Apple Silicon (MPS)"
                s.vram_used_gb = torch.mps.current_allocated_memory() / (1024 ** 3) if hasattr(torch.mps, "current_allocated_memory") else 0
        except (ImportError, Exception):
            pass

        self.samples.append(s)
        return s

    def check_threshold(self, sample: Optional[Sample] = None) -> Optional[str]:
        """Check if VRAM usage exceeds threshold. Returns warning or None."""
        s = sample or (self.samples[-1] if self.samples else self.sample())
        if s.vram_total_gb > 0:
            usage = s.vram_used_gb / s.vram_total_gb
            if usage > self.threshold:
                return f"VRAM at {usage:.0%} ({s.vram_used_gb:.1f}/{s.vram_total_gb:.0f}GB) — above {self.threshold:.0%} threshold"
        return None
