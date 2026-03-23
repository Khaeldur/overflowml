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
    EXPERT_OFFLOAD = "expert_offload"   # MoE: shared layers on GPU, experts swapped from RAM
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
class MoEProfile:
    """Mixture-of-Experts model characteristics."""
    total_params_b: float = 0.0       # total parameters in billions
    active_params_b: float = 0.0      # active parameters per token in billions
    num_experts: int = 0              # total expert count per MoE layer
    num_active_experts: int = 0       # experts activated per token (routed + shared)
    shared_layers_gb: float = 0.0     # size of non-expert layers (attention, embed, router) in GB
    expert_size_gb: float = 0.0       # size of all expert FFN weights in GB

    @property
    def sparsity_ratio(self) -> float:
        if self.total_params_b == 0:
            return 0.0
        return 1.0 - (self.active_params_b / self.total_params_b)

    @property
    def active_expert_gb(self) -> float:
        if self.num_experts == 0:
            return 0.0
        return self.expert_size_gb * (self.num_active_experts / self.num_experts)

    @property
    def gpu_footprint_gb(self) -> float:
        return self.shared_layers_gb + self.active_expert_gb


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
    moe: Optional[MoEProfile] = None  # set for MoE models
    llamacpp_flags: list[str] = field(default_factory=list)
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
    moe: Optional[MoEProfile] = None,
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

    # --- MoE expert offload path
    if moe is not None:
        return _pick_moe_strategy(hw, model_size_gb, moe, prefer_speed, allow_quantization,
                                  force_offload, force_quant)

    # --- Dtype selection
    if hw.supports_bf16:
        s.dtype = "bfloat16"
    elif hw.accelerator in (Accelerator.CUDA, Accelerator.ROCm, Accelerator.MPS, Accelerator.MLX):
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


def _pick_moe_strategy(
    hw: HardwareProfile,
    model_size_gb: float,
    moe: MoEProfile,
    prefer_speed: bool,
    allow_quantization: bool,
    force_offload: Optional[OffloadMode],
    force_quant: Optional[QuantMode],
) -> Strategy:
    """Strategy selection for Mixture-of-Experts models.

    MoE models only activate a fraction of total params per token.
    Key insight: shared layers (attention, embeddings, router) stay on GPU,
    expert FFN weights can live in RAM and swap on demand.
    """
    s = Strategy(moe=moe)
    vram = hw.gpu_vram_gb
    ram = hw.system_ram_gb

    # Dtype
    if hw.supports_bf16:
        s.dtype = "bfloat16"
    elif hw.accelerator in (Accelerator.CUDA, Accelerator.ROCm, Accelerator.MPS, Accelerator.MLX):
        s.dtype = "float16"
    else:
        s.dtype = "float32"

    if force_offload:
        s.offload = force_offload
    if force_quant:
        s.quantization = force_quant
    if force_offload or force_quant:
        s.estimated_vram_gb = moe.gpu_footprint_gb
        return s

    # Can the FULL model fit in VRAM? (small MoE models)
    if model_size_gb * 1.15 <= vram:
        s.offload = OffloadMode.NONE
        s.estimated_vram_gb = model_size_gb * 1.15
        s.notes.append(f"Full MoE model ({model_size_gb:.0f}GB) fits in VRAM")
        if prefer_speed:
            s.compile = True
        return s

    # Expert offload: shared layers on GPU, experts swapped from RAM
    # This is the sweet spot for large MoE models on consumer hardware
    gpu_needed = moe.gpu_footprint_gb * 1.2  # 20% headroom for KV cache
    ram_needed = model_size_gb * 1.1  # full model in RAM + 10% overhead

    if gpu_needed <= vram and ram_needed <= ram:
        s.offload = OffloadMode.EXPERT_OFFLOAD
        s.estimated_vram_gb = gpu_needed
        s.notes.append(
            f"MoE expert offload: {moe.shared_layers_gb:.0f}GB shared layers + "
            f"{moe.active_expert_gb:.0f}GB active experts on GPU"
        )
        s.notes.append(
            f"Inactive experts ({moe.expert_size_gb - moe.active_expert_gb:.0f}GB) "
            f"swap from {ram:.0f}GB RAM"
        )
        s.notes.append(f"Sparsity: {moe.sparsity_ratio:.0%} — only {moe.active_params_b:.0f}B of "
                        f"{moe.total_params_b:.0f}B params active per token")
        # llama.cpp flags for MoE offload
        ngl = _estimate_ngl_moe(moe, vram)
        s.llamacpp_flags = [f"-ngl {ngl}", f"-c 8192", "--mlock"]
        return s

    # Expert offload with quantization
    if allow_quantization and ram_needed * 0.5 <= ram:
        s.offload = OffloadMode.EXPERT_OFFLOAD
        s.quantization = QuantMode.INT4
        q4_total = model_size_gb * 0.3
        q4_gpu = moe.gpu_footprint_gb * 0.3 * 1.2
        s.estimated_vram_gb = q4_gpu
        s.notes.append(f"MoE expert offload + INT4: {q4_total:.0f}GB total in RAM, "
                        f"{q4_gpu:.0f}GB active on GPU")
        ngl = _estimate_ngl_moe(moe, vram)
        s.llamacpp_flags = [f"-ngl {ngl}", f"-c 8192", "--mlock"]
        return s

    # Fallback: sequential offload (worst case)
    if ram >= model_size_gb * 0.5:
        s.offload = OffloadMode.SEQUENTIAL_CPU
        s.estimated_vram_gb = 3.0
        s.gc_between_steps = True
        s.warnings.append(f"MoE model too large for expert offload — falling back to sequential")
        return s

    s.offload = OffloadMode.DISK
    s.estimated_vram_gb = 3.0
    s.warnings.append(f"Insufficient memory for {model_size_gb:.0f}GB MoE model")
    return s


def _estimate_ngl_moe(moe: MoEProfile, vram_gb: float) -> int:
    """Estimate how many layers to offload to GPU for MoE via llama.cpp."""
    if moe.shared_layers_gb == 0:
        return 99  # let llama.cpp figure it out
    # Reserve 2GB for KV cache, rest for layers
    available = vram_gb - 2.0
    if available <= 0:
        return 0
    # For MoE, each layer's shared part (attention) is small,
    # but expert FFNs are large. Use 99 and let llama.cpp auto-split.
    return 99


def plan_llamacpp(
    model_path: str,
    moe: Optional[MoEProfile] = None,
    hw: Optional[HardwareProfile] = None,
    context_size: int = 8192,
    port: int = 8080,
) -> dict:
    """Generate llama.cpp launch configuration for a GGUF model.

    Returns a dict with 'command', 'flags', and 'notes'.
    """
    if hw is None:
        hw = detect_hardware()

    vram = hw.gpu_vram_gb
    ram = hw.system_ram_gb

    flags = [f"-m {model_path}", f"-c {context_size}", f"--port {port}"]

    if moe:
        # MoE: use max GPU layers, llama.cpp handles expert routing
        flags.append("-ngl 99")
        flags.append("--mlock")
        notes = [
            f"MoE model: {moe.total_params_b:.0f}B total, {moe.active_params_b:.0f}B active",
            f"Expert offload: shared layers pinned to GPU ({vram:.0f}GB), "
            f"experts swap from RAM ({ram:.0f}GB)",
            f"Expect ~10-25 tok/s depending on expert cache hit rate",
        ]
    else:
        # Dense model
        ngl = min(99, int((vram - 2) / 0.5))  # rough: ~0.5GB per layer
        flags.append(f"-ngl {ngl}")
        notes = [f"Dense model with {ngl} layers on GPU"]

    return {
        "command": "llama-server " + " ".join(flags),
        "flags": flags,
        "notes": notes,
    }


MOE_REGISTRY: dict[str, MoEProfile] = {
    "qwen3.5-397b": MoEProfile(
        total_params_b=397, active_params_b=17, num_experts=512, num_active_experts=11,
        shared_layers_gb=0, expert_size_gb=0,  # filled at runtime based on quant size
    ),
    "qwen3.5-122b": MoEProfile(
        total_params_b=122, active_params_b=10, num_experts=128, num_active_experts=9,
        shared_layers_gb=0, expert_size_gb=0,
    ),
    "qwen3.5-35b": MoEProfile(
        total_params_b=35, active_params_b=3, num_experts=256, num_active_experts=9,
        shared_layers_gb=0, expert_size_gb=0,
    ),
    "minimax-m2.5": MoEProfile(
        total_params_b=230, active_params_b=10, num_experts=256, num_active_experts=8,
        shared_layers_gb=0, expert_size_gb=0,
    ),
    "step-3.5-flash": MoEProfile(
        total_params_b=196, active_params_b=11, num_experts=64, num_active_experts=8,
        shared_layers_gb=0, expert_size_gb=0,
    ),
    "mimo-v2-flash": MoEProfile(
        total_params_b=309, active_params_b=15, num_experts=128, num_active_experts=8,
        shared_layers_gb=0, expert_size_gb=0,
    ),
    "deepseek-v3.2": MoEProfile(
        total_params_b=685, active_params_b=37, num_experts=256, num_active_experts=8,
        shared_layers_gb=0, expert_size_gb=0,
    ),
    "glm-5": MoEProfile(
        total_params_b=744, active_params_b=40, num_experts=256, num_active_experts=8,
        shared_layers_gb=0, expert_size_gb=0,
    ),
    "kimi-k2.5": MoEProfile(
        total_params_b=1000, active_params_b=32, num_experts=384, num_active_experts=8,
        shared_layers_gb=0, expert_size_gb=0,
    ),
    "mixtral-8x7b": MoEProfile(
        total_params_b=46.7, active_params_b=12.9, num_experts=8, num_active_experts=2,
        shared_layers_gb=0, expert_size_gb=0,
    ),
    "mixtral-8x22b": MoEProfile(
        total_params_b=141, active_params_b=39, num_experts=8, num_active_experts=2,
        shared_layers_gb=0, expert_size_gb=0,
    ),
    "nemotron-3-super": MoEProfile(
        total_params_b=120, active_params_b=12, num_experts=128, num_active_experts=8,
        shared_layers_gb=0, expert_size_gb=0,
    ),
}


def get_moe_profile(model_name: str, model_size_gb: float) -> Optional[MoEProfile]:
    """Look up a known MoE profile by model name and fill in size estimates."""
    name_lower = model_name.lower()
    for key, profile in MOE_REGISTRY.items():
        if key in name_lower:
            p = MoEProfile(
                total_params_b=profile.total_params_b,
                active_params_b=profile.active_params_b,
                num_experts=profile.num_experts,
                num_active_experts=profile.num_active_experts,
                shared_layers_gb=model_size_gb * 0.30,
                expert_size_gb=model_size_gb * 0.70,
            )
            return p
    return None


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
    elif s.offload == OffloadMode.EXPERT_OFFLOAD and s.moe:
        s.estimated_vram_gb = s.moe.gpu_footprint_gb
    else:
        s.estimated_vram_gb = size * 1.15
