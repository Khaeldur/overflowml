"""Planning engine — computes viable strategies and recommends one."""

from __future__ import annotations

import logging
from typing import Optional, Union

from .types import HardwareInfo, ModelInfo, PlanResult, StrategyCandidate
from .hardware import detect_hardware_info, hardware_info_to_legacy
from .explain import build_explanation

logger = logging.getLogger("overflowml")

SPEED_TIERS = {
    ("none", "none", "none"): "fastest",
    ("fp8", "none", "none"): "fastest",
    ("none", "none", "auto"): "fast",
    ("fp8", "none", "auto"): "fast",
    ("none", "layer_hybrid", "none"): "fast",
    ("fp8", "layer_hybrid", "none"): "fast",
    ("int4", "layer_hybrid", "none"): "medium",
    ("none", "model_cpu", "none"): "medium",
    ("fp8", "model_cpu", "none"): "medium",
    ("none", "model_cpu", "auto"): "medium",
    ("none", "sequential_cpu", "none"): "slow",
    ("int4", "sequential_cpu", "none"): "slowest",
    ("none", "disk", "none"): "slowest",
    ("int4", "disk", "none"): "slowest",
}

QUALITY_RISK = {
    "none": "none",
    "fp8": "minimal",
    "int8": "low",
    "int4": "medium",
    "gguf": "medium",
}


def plan(
    model_or_size: Union[str, float],
    hw: Optional[HardwareInfo] = None,
    *,
    compare: bool = False,
    trust_remote_code: bool = False,
    lora_size_gb: Optional[float] = None,
) -> PlanResult:
    """Plan optimal loading strategy for a model.

    Args:
        model_or_size: HuggingFace model ID (str) or size in GB (float)
        hw: Hardware info. Auto-detected if None.
        compare: If True, return all viable strategies, not just recommended.
        trust_remote_code: Passed to HF model inspection.

    Returns:
        PlanResult with recommended strategy, all candidates, and explanation.
    """
    result = PlanResult()

    # Detect hardware
    if hw is None:
        hw = detect_hardware_info()
    result.hardware = hw

    # Resolve model size
    model_info = None
    if isinstance(model_or_size, str):
        try:
            model_size_gb = float(model_or_size)
        except ValueError:
            from ..inspect import inspect_model
            model_info = inspect_model(model_or_size, trust_remote_code)
            result.model = model_info
            if model_info.estimated_sizes_gb and "fp16" in model_info.estimated_sizes_gb:
                model_size_gb = model_info.estimated_sizes_gb["fp16"]
            else:
                model_size_gb = 14.0
                logger.warning("Could not estimate size for %s — using 14GB default", model_or_size)
    else:
        model_size_gb = float(model_or_size)

    if model_info is None:
        model_info = ModelInfo(model_id=str(model_size_gb), source="user-provided")
        result.model = model_info

    # Apply LoRA overhead
    if lora_size_gb and lora_size_gb > 0:
        model_size_gb += lora_size_gb
        if model_info:
            model_info.notes.append(f"LoRA adapter adds {lora_size_gb:.1f}GB → effective size {model_size_gb:.1f}GB")

    # Convert to legacy hw for pick_strategy
    legacy_hw = hardware_info_to_legacy(hw)

    # Generate candidates
    candidates = _generate_candidates(legacy_hw, model_size_gb)
    result.strategies = candidates

    # Pick recommended
    viable = [c for c in candidates if c.viable]
    if viable:
        viable[0].recommended = True
        result.recommended = viable[0]

    # Build explanation
    result.explanation = build_explanation(
        model_size_gb, hw, result.recommended, candidates, model_info
    )

    # If not compare mode, trim to just recommended
    if not compare:
        result.strategies = [c for c in candidates if c.viable]

    return result


def _generate_candidates(legacy_hw, model_size_gb: float) -> list[StrategyCandidate]:
    """Generate all candidate strategies by calling pick_strategy with different params."""
    from ..strategy import DistributionMode, OffloadMode, QuantMode, pick_strategy

    candidates = []
    seen = set()

    configs = [
        {"prefer_speed": True, "allow_quantization": True},
        {"prefer_speed": False, "allow_quantization": True},
        {"prefer_speed": True, "allow_quantization": False},
        {"prefer_speed": False, "allow_quantization": False},
    ]

    # Force specific offload modes
    for mode in [OffloadMode.NONE, OffloadMode.MODEL_CPU, OffloadMode.LAYER_HYBRID, OffloadMode.SEQUENTIAL_CPU]:
        configs.append({"force_offload": mode})

    # Force quantization modes
    if legacy_hw.supports_fp8:
        configs.append({"force_quant": QuantMode.FP8})
    configs.append({"force_quant": QuantMode.INT4})

    for cfg in configs:
        try:
            s = pick_strategy(legacy_hw, model_size_gb, **cfg)
        except Exception:
            continue

        key = (s.quantization.value, s.offload.value, s.distribution.value)
        if key in seen:
            continue
        seen.add(key)

        candidate = _strategy_to_candidate(s, model_size_gb, legacy_hw)
        candidates.append(candidate)

    # Sort: viable first, then by speed tier
    speed_order = {"fastest": 0, "fast": 1, "medium": 2, "slow": 3, "slowest": 4, "unknown": 5}
    candidates.sort(key=lambda c: (not c.viable, speed_order.get(c.estimated_speed, 5)))

    return candidates


def _strategy_to_candidate(s, model_size_gb: float, legacy_hw) -> StrategyCandidate:
    """Convert a legacy Strategy to a StrategyCandidate."""
    from ..strategy import DistributionMode, OffloadMode, QuantMode

    # Build name
    parts = []
    if s.quantization != QuantMode.NONE:
        parts.append(s.quantization.value.upper())
    if s.distribution != DistributionMode.NONE:
        parts.append(f"multi-GPU ({s.num_gpus_used})")
    if s.offload != OffloadMode.NONE:
        parts.append(s.offload.value.replace("_", " "))
    name = "fp16 " + " + ".join(parts) if parts else "fp16 direct load"

    # Speed tier
    key = (s.quantization.value, s.offload.value, s.distribution.value)
    speed = SPEED_TIERS.get(key, "unknown")

    # Quality risk
    quality = QUALITY_RISK.get(s.quantization.value, "unknown")

    # Viability + gotcha-aware rejection reasons
    viable = True
    rejection = ""
    if s.offload == OffloadMode.DISK:
        viable = False
        rejection = "requires disk offload (insufficient memory for VRAM + RAM)"
    elif s.offload == OffloadMode.NONE and s.estimated_vram_gb > legacy_hw.gpu_vram_gb * 1.05:
        viable = False
        rejection = f"exceeds VRAM ({s.estimated_vram_gb:.1f}GB > {legacy_hw.gpu_vram_gb:.0f}GB)"

    # Gotcha-aware notes
    notes = list(s.notes)
    if s.quantization == QuantMode.FP8 and s.offload != OffloadMode.NONE:
        if legacy_hw.os == "Windows":
            notes.append("FP8 + CPU offload crashes on Windows (Float8Tensor device mismatch) — avoided")
    if s.offload == OffloadMode.SEQUENTIAL_CPU:
        notes.append("attention_slicing disabled — crashes with sequential offload (CUDA illegal memory access)")
    if s.quantization == QuantMode.FP8 and not legacy_hw.supports_fp8:
        viable = False
        rejection = "FP8 requires torchao — not installed"
    if legacy_hw.os == "Windows" and hasattr(s, 'offload') and s.offload != OffloadMode.NONE:
        notes.append("expandable_segments disabled (not supported on Windows WDDM)")

    # Warnings from strategy become rejection reasons for non-viable
    for w in s.warnings:
        if not rejection and ("insufficient" in w.lower() or "exceeds" in w.lower()):
            viable = False
            rejection = w

    # Estimated RAM
    if s.offload in (OffloadMode.SEQUENTIAL_CPU, OffloadMode.MODEL_CPU):
        est_ram = model_size_gb * 1.3
    else:
        est_ram = max(model_size_gb * 0.1, 2.0)

    return StrategyCandidate(
        name=name,
        viable=viable,
        estimated_vram_gb=s.estimated_vram_gb,
        estimated_ram_gb=est_ram,
        estimated_speed=speed,
        quality_risk=quality,
        rejection_reason=rejection,
        notes=notes,
    )
