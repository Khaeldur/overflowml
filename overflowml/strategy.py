"""Strategy engine — decides how to load a model given hardware constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .detect import Accelerator, HardwareProfile


class OffloadMode(Enum):
    NONE = "none"                       # model fits in VRAM
    MODEL_CPU = "model_cpu"             # full components moved to/from GPU
    SEQUENTIAL_CPU = "sequential_cpu"   # one layer at a time (lowest VRAM)
    DISK = "disk"                       # offload to disk (for very large models)


class QuantMode(Enum):
    NONE = "none"           # full precision (BF16/FP16/FP32)
    FP8 = "fp8"             # 8-bit float (halves memory)
    INT8 = "int8"           # 8-bit integer
    INT4 = "int4"           # 4-bit integer (quarter memory)
    GGUF = "gguf"           # llama.cpp quantization


class DistributionMode(Enum):
    NONE = "none"                  # Single GPU
    DEVICE_MAP_AUTO = "auto"       # accelerate distributes layers across GPUs


@dataclass
class Strategy:
    offload: OffloadMode = OffloadMode.NONE
    quantization: QuantMode = QuantMode.NONE
    distribution: DistributionMode = DistributionMode.NONE
    num_gpus_used: int = 1
    dtype: str = "bfloat16"
    compile: bool = False
    compile_mode: str = "max-autotune"
    attention_slicing: bool = False
    vae_slicing: bool = False
    gc_between_steps: bool = False
    vram_threshold: float = 0.7       # trigger cleanup at this % of VRAM
    estimated_vram_gb: float = 0.0
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = []
        if self.distribution != DistributionMode.NONE:
            parts.append(f"Distribution: {self.distribution.value} ({self.num_gpus_used} GPUs)")
        if self.quantization != QuantMode.NONE:
            parts.append(f"Quantization: {self.quantization.value}")
        parts.append(f"Offload: {self.offload.value}")
        parts.append(f"Dtype: {self.dtype}")
        if self.compile:
            parts.append(f"torch.compile: {self.compile_mode}")
        if self.gc_between_steps:
            parts.append(f"GC cleanup: enabled (threshold {self.vram_threshold:.0%})")
        if self.estimated_vram_gb > 0:
            parts.append(f"Estimated peak VRAM: {self.estimated_vram_gb:.1f}GB")
        for w in self.warnings:
            parts.append(f"WARNING: {w}")
        for n in self.notes:
            parts.append(f"  - {n}")
        return "\n".join(parts)


def pick_strategy(
    hw: HardwareProfile,
    model_size_gb: float,
    *,
    prefer_speed: bool = True,
    allow_quantization: bool = True,
    force_offload: Optional[OffloadMode] = None,
    force_quant: Optional[QuantMode] = None,
) -> Strategy:
    """Pick the optimal loading strategy for a model on this hardware.

    Args:
        hw: Hardware profile from detect_hardware()
        model_size_gb: Model size in GB (BF16/FP16 weights)
        prefer_speed: If True, prefer faster strategies over lower VRAM
        allow_quantization: If True, allow FP8/INT8/INT4 quantization
        force_offload: Override the offload mode
        force_quant: Override the quantization mode
    """
    s = Strategy()

    # --- Dtype selection
    if hw.supports_bf16:
        s.dtype = "bfloat16"
    elif hw.accelerator in (Accelerator.CUDA, Accelerator.MPS, Accelerator.MLX):
        s.dtype = "float16"
    else:
        s.dtype = "float32"

    # --- Apple Silicon (unified memory)
    if hw.unified_memory:
        effective = hw.effective_memory_gb
        if model_size_gb <= effective:
            s.offload = OffloadMode.NONE
            s.estimated_vram_gb = model_size_gb
            s.notes.append(f"Model ({model_size_gb:.0f}GB) fits in unified memory ({effective:.0f}GB)")
        else:
            s.offload = OffloadMode.SEQUENTIAL_CPU
            s.estimated_vram_gb = model_size_gb * 0.05  # ~5% at a time
            s.warnings.append(f"Model ({model_size_gb:.0f}GB) exceeds effective memory ({effective:.0f}GB) — will be slow")

        if hw.accelerator == Accelerator.MLX:
            s.notes.append("MLX uses lazy evaluation — memory usage is dynamic")
        return s

    # --- CUDA / discrete GPU
    vram = hw.gpu_vram_gb
    ram = hw.system_ram_gb

    # Handle forced modes
    if force_offload:
        s.offload = force_offload
    if force_quant:
        s.quantization = force_quant

    if force_offload or force_quant:
        _estimate_vram(s, model_size_gb, vram)
        return s

    # Can the model fit in VRAM at full precision?
    if model_size_gb * 1.15 <= vram:  # 15% headroom for activations
        s.offload = OffloadMode.NONE
        s.estimated_vram_gb = model_size_gb * 1.15
        s.notes.append(f"Model fits in VRAM ({model_size_gb:.0f}GB + activations < {vram:.0f}GB)")
        if prefer_speed:
            s.compile = True
        return s

    # Can FP8 make it fit on single GPU?
    fp8_size = model_size_gb * 0.55  # FP8 ≈ 55% of BF16
    if allow_quantization and hw.supports_fp8 and fp8_size * 1.15 <= vram:
        s.quantization = QuantMode.FP8
        s.offload = OffloadMode.NONE
        s.estimated_vram_gb = fp8_size * 1.15
        s.notes.append(f"FP8 reduces model to ~{fp8_size:.0f}GB — fits in VRAM")
        if prefer_speed:
            s.compile = True
        return s

    # --- Multi-GPU distribution (before CPU offload) ---
    if hw.num_gpus > 1:
        total_vram = hw.total_gpu_vram_gb

        # Model fits across all GPUs at full precision?
        if model_size_gb * 1.15 <= total_vram:
            s.distribution = DistributionMode.DEVICE_MAP_AUTO
            s.num_gpus_used = hw.num_gpus
            s.estimated_vram_gb = model_size_gb * 1.15 / hw.num_gpus
            s.notes.append(f"Distributed across {hw.num_gpus} GPUs ({vram:.0f}GB each)")
            if prefer_speed:
                s.compile = True
            return s

        # Model fits across GPUs with FP8?
        if allow_quantization and hw.supports_fp8 and fp8_size * 1.15 <= total_vram:
            s.quantization = QuantMode.FP8
            s.distribution = DistributionMode.DEVICE_MAP_AUTO
            s.num_gpus_used = hw.num_gpus
            s.estimated_vram_gb = fp8_size * 1.15 / hw.num_gpus
            s.notes.append(f"FP8 + distributed across {hw.num_gpus} GPUs")
            if prefer_speed:
                s.compile = True
            return s

        # Multi-GPU + CPU offload
        if ram >= model_size_gb * 1.3:
            s.distribution = DistributionMode.DEVICE_MAP_AUTO
            s.num_gpus_used = hw.num_gpus
            s.offload = OffloadMode.MODEL_CPU
            s.estimated_vram_gb = model_size_gb * 0.7 / hw.num_gpus
            s.gc_between_steps = True
            s.notes.append(f"Distributed + CPU offload across {hw.num_gpus} GPUs + RAM")
            return s

    # Need CPU offload — but can model_cpu_offload handle it?
    if model_size_gb <= vram * 1.5 and ram >= model_size_gb * 1.3:
        # Model components individually fit, just not all at once
        if allow_quantization and hw.supports_fp8:
            s.quantization = QuantMode.FP8
            s.offload = OffloadMode.MODEL_CPU
            s.estimated_vram_gb = fp8_size * 0.7
            s.notes.append("FP8 + model offload: components moved to GPU one at a time")
        else:
            s.offload = OffloadMode.MODEL_CPU
            s.estimated_vram_gb = model_size_gb * 0.7
            s.notes.append("Model offload: components moved to GPU one at a time")
        s.gc_between_steps = True
        return s

    # Model too large even for component offload — need sequential
    if ram >= model_size_gb * 1.3:
        s.offload = OffloadMode.SEQUENTIAL_CPU
        s.estimated_vram_gb = 3.0  # ~1 layer at a time
        s.gc_between_steps = True
        s.notes.append(f"Sequential offload: 1 layer at a time (~3GB VRAM), model lives in {ram:.0f}GB RAM")

        # FP8 is INCOMPATIBLE with CPU offload (Float8Tensor can't move between devices)
        if hw.os == "Windows":
            s.warnings.append("FP8 incompatible with CPU offload on Windows (torchao Float8Tensor device mismatch)")
        elif allow_quantization and hw.supports_fp8:
            # On Linux, torchao may support FP8 + offload — experimental
            s.notes.append("FP8 + sequential offload: experimental on Linux")

        # attention_slicing conflicts with sequential offload
        s.warnings.append("Do NOT enable attention_slicing with sequential offload (causes CUDA illegal memory access)")
        return s

    # Extreme case: not enough RAM either
    if allow_quantization:
        s.quantization = QuantMode.INT4
        s.offload = OffloadMode.SEQUENTIAL_CPU
        s.estimated_vram_gb = 3.0
        s.gc_between_steps = True
        int4_size = model_size_gb * 0.3
        s.warnings.append(f"Model ({model_size_gb:.0f}GB) exceeds both VRAM and RAM — INT4 quantization to ~{int4_size:.0f}GB")
        return s

    s.offload = OffloadMode.DISK
    s.estimated_vram_gb = 3.0
    s.gc_between_steps = True
    s.warnings.append(f"Insufficient memory. Model: {model_size_gb:.0f}GB, VRAM: {vram:.0f}GB, RAM: {ram:.0f}GB — disk offload required (very slow)")
    return s


def _estimate_vram(s: Strategy, model_gb: float, vram_gb: float):
    """Fill in estimated VRAM for forced strategies."""
    size = model_gb
    if s.quantization == QuantMode.FP8:
        size *= 0.55
    elif s.quantization == QuantMode.INT8:
        size *= 0.55
    elif s.quantization == QuantMode.INT4:
        size *= 0.3

    if s.offload == OffloadMode.SEQUENTIAL_CPU:
        s.estimated_vram_gb = 3.0
    elif s.offload == OffloadMode.MODEL_CPU:
        s.estimated_vram_gb = size * 0.7
    else:
        s.estimated_vram_gb = size * 1.15
