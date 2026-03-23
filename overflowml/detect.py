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
    # Multi-GPU fields
    num_gpus: int = 1
    gpu_names: list[str] = field(default_factory=list)
    gpu_vram_gbs: list[float] = field(default_factory=list)
    total_gpu_vram_gb: float = 0.0

    def __post_init__(self):
        if self.total_gpu_vram_gb == 0.0 and self.gpu_vram_gb > 0:
            self.total_gpu_vram_gb = self.gpu_vram_gb

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
        return self.total_gpu_vram_gb + self.overflow_gb

    def summary(self) -> str:
        lines = [f"Accelerator: {self.accelerator.value}"]
        if self.num_gpus > 1:
            lines.append(f"GPUs: {self.num_gpus}x {self.gpu_name} ({self.gpu_vram_gb:.0f}GB each, {self.total_gpu_vram_gb:.0f}GB total)")
        elif self.gpu_name:
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

        num_gpus = torch.cuda.device_count()
        gpu_names = []
        gpu_vram_gbs = []
        compute_caps = []

        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            gpu_names.append(props.name)
            gpu_vram_gbs.append(props.total_memory / (1024 ** 3))
            compute_caps.append((props.major, props.minor))

        # Use first GPU for primary fields (backward compat)
        # gpu_vram_gb = smallest GPU (conservative for balanced allocation)
        primary_cc = compute_caps[0]
        min_vram = min(gpu_vram_gbs)

        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)

        is_rocm = getattr(torch.version, "hip", None) is not None
        accelerator = Accelerator.ROCm if is_rocm else Accelerator.CUDA

        supports_fp8 = False
        if not is_rocm:
            try:
                from torchao.quantization import Float8WeightOnlyConfig  # noqa: F401
                supports_fp8 = True
            except ImportError:
                pass

        return HardwareProfile(
            accelerator=accelerator,
            gpu_name=gpu_names[0],
            gpu_vram_gb=min_vram,
            gpu_compute_capability=primary_cc,
            system_ram_gb=ram_gb,
            unified_memory=False,
            os=platform.system(),
            cpu_cores=psutil.cpu_count(logical=False) or 1,
            supports_bf16=all(cc >= (8, 0) for cc in compute_caps),
            supports_fp8=supports_fp8,
            num_gpus=num_gpus,
            gpu_names=gpu_names,
            gpu_vram_gbs=gpu_vram_gbs,
            total_gpu_vram_gb=sum(gpu_vram_gbs),
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
            import subprocess
            result = subprocess.run(
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
