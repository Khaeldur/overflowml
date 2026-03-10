"""Hardware detection — GPU, CPU, RAM, platform."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Accelerator(Enum):
    CUDA = "cuda"
    MPS = "mps"         # Apple Metal Performance Shaders
    MLX = "mlx"         # Apple MLX framework
    ROCm = "rocm"       # AMD GPUs
    CPU = "cpu"


@dataclass
class HardwareProfile:
    accelerator: Accelerator
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    gpu_compute_capability: tuple[int, int] = (0, 0)
    system_ram_gb: float = 0.0
    unified_memory: bool = False  # Apple Silicon shares RAM/VRAM
    os: str = ""
    cpu_cores: int = 0
    supports_bf16: bool = False
    supports_fp8: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def overflow_gb(self) -> float:
        """How much a model can overflow GPU into system RAM."""
        if self.unified_memory:
            return 0.0  # already shared
        return max(0, self.system_ram_gb - 16)  # keep 16GB for OS/apps

    @property
    def effective_memory_gb(self) -> float:
        """Total memory available for model loading."""
        if self.unified_memory:
            return self.system_ram_gb * 0.75  # macOS reserves ~25%
        return self.gpu_vram_gb + self.overflow_gb

    def summary(self) -> str:
        lines = [f"Accelerator: {self.accelerator.value}"]
        if self.gpu_name:
            lines.append(f"GPU: {self.gpu_name} ({self.gpu_vram_gb:.0f}GB VRAM)")
        lines.append(f"System RAM: {self.system_ram_gb:.0f}GB")
        if self.unified_memory:
            lines.append(f"Unified memory: {self.effective_memory_gb:.0f}GB effective")
        else:
            lines.append(f"Overflow capacity: {self.overflow_gb:.0f}GB (total effective: {self.effective_memory_gb:.0f}GB)")
        lines.append(f"BF16: {'yes' if self.supports_bf16 else 'no'} | FP8: {'yes' if self.supports_fp8 else 'no'}")
        return "\n".join(lines)


def _detect_cuda() -> Optional[HardwareProfile]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None

        props = torch.cuda.get_device_properties(0)
        cc = (props.major, props.minor)

        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)

        supports_fp8 = False
        try:
            from torchao.quantization import Float8WeightOnlyConfig  # noqa: F401
            supports_fp8 = True
        except ImportError:
            pass

        return HardwareProfile(
            accelerator=Accelerator.CUDA,
            gpu_name=props.name,
            gpu_vram_gb=props.total_memory / (1024 ** 3),
            gpu_compute_capability=cc,
            system_ram_gb=ram_gb,
            unified_memory=False,
            os=platform.system(),
            cpu_cores=psutil.cpu_count(logical=False) or 1,
            supports_bf16=cc >= (8, 0),  # Ampere+
            supports_fp8=supports_fp8,
        )
    except ImportError:
        return None


def _detect_mps() -> Optional[HardwareProfile]:
    try:
        import torch
        if not torch.backends.mps.is_available():
            return None

        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)

        gpu_name = "Apple Silicon"
        try:
            result = __import__("subprocess").run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                gpu_name = result.stdout.strip()
        except Exception:
            pass

        return HardwareProfile(
            accelerator=Accelerator.MPS,
            gpu_name=gpu_name,
            gpu_vram_gb=ram_gb,  # unified — all RAM is "VRAM"
            system_ram_gb=ram_gb,
            unified_memory=True,
            os="Darwin",
            cpu_cores=psutil.cpu_count(logical=False) or 1,
            supports_bf16=True,  # all Apple Silicon supports BF16
            supports_fp8=False,
        )
    except ImportError:
        return None


def _detect_mlx() -> Optional[HardwareProfile]:
    try:
        import mlx.core as mx

        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)

        return HardwareProfile(
            accelerator=Accelerator.MLX,
            gpu_name="Apple Silicon (MLX)",
            gpu_vram_gb=ram_gb,
            system_ram_gb=ram_gb,
            unified_memory=True,
            os="Darwin",
            cpu_cores=psutil.cpu_count(logical=False) or 1,
            supports_bf16=True,
            supports_fp8=False,
            extra={"mlx_version": mx.__version__ if hasattr(mx, "__version__") else "unknown"},
        )
    except ImportError:
        return None


def detect_hardware(prefer_mlx: bool = False) -> HardwareProfile:
    """Auto-detect the best available hardware.

    Priority: CUDA > MLX > MPS > CPU
    Set prefer_mlx=True on macOS to prefer MLX over MPS.
    """
    if not prefer_mlx:
        cuda = _detect_cuda()
        if cuda:
            return cuda

    if sys.platform == "darwin":
        if prefer_mlx:
            mlx = _detect_mlx()
            if mlx:
                return mlx

        mps = _detect_mps()
        if mps:
            return mps

        if not prefer_mlx:
            mlx = _detect_mlx()
            if mlx:
                return mlx

    cuda = _detect_cuda()
    if cuda:
        return cuda

    import psutil
    return HardwareProfile(
        accelerator=Accelerator.CPU,
        gpu_name="None",
        system_ram_gb=psutil.virtual_memory().total / (1024 ** 3),
        os=platform.system(),
        cpu_cores=psutil.cpu_count(logical=False) or 1,
    )
