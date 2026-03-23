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

    # Runtime intelligence: KV cache, flash attention, PCIe, fragmentation
    _add_runtime_intelligence(result, model_size_gb, hw)

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


def _add_runtime_intelligence(result: PlanResult, model_size_gb: float, hw: HardwareInfo):
    """Add runtime analysis: KV cache, flash attention, PCIe, fragmentation, load time."""
    from .runtime import (
        context_adjusted_vram,
        detect_flash_attention,
        detect_pcie_bandwidth,
        diagnose_fragmentation,
        estimate_load_time,
        suggest_draft_model,
    )

    runtime_notes = []

    # KV cache warning — the hidden VRAM killer
    ctx_vram = context_adjusted_vram(model_size_gb, context_length=4096)
    if ctx_vram["kv_cache_gb"] > 1.0:
        runtime_notes.append(
            f"KV cache warning: {ctx_vram['breakdown']}"
        )
        if ctx_vram["total_gb"] > hw.total_vram_gb and hw.total_vram_gb > 0:
            runtime_notes.append(
                f"At 4K context, total VRAM need ({ctx_vram['total_gb']:.0f}GB) "
                f"exceeds GPU ({hw.total_vram_gb:.0f}GB) — reduce context or use longer offload"
            )

    # Flash attention
    flash = detect_flash_attention()
    if not flash.available:
        runtime_notes.append("No efficient attention backend — using naive attention (2-4x slower, higher VRAM)")
        runtime_notes.append("Fix: pip install flash-attn or upgrade torch >= 2.0")
    else:
        runtime_notes.append(f"Attention backend: {flash.backend} ({flash.notes[0] if flash.notes else ''})")

    # PCIe bandwidth (relevant for hybrid/offload)
    if result.recommended and "hybrid" in (result.recommended.name or "").lower():
        pcie = detect_pcie_bandwidth()
        if pcie.detected:
            ram_portion = model_size_gb - (hw.total_vram_gb * 0.9 if hw.gpus else 0)
            if ram_portion > 0 and pcie.practical_gbps > 0:
                overhead_per_pass = ram_portion / pcie.practical_gbps
                runtime_notes.append(
                    f"PCIe Gen{pcie.generation} x{pcie.width}: ~{overhead_per_pass:.1f}s transfer overhead per forward pass "
                    f"({ram_portion:.0f}GB over {pcie.practical_gbps:.0f} GB/s)"
                )

    # Fragmentation check
    frag = diagnose_fragmentation()
    if frag.is_fragmented:
        runtime_notes.extend(frag.notes)

    # Load time estimate
    load_est = estimate_load_time(model_size_gb)
    if model_size_gb > 20:
        runtime_notes.append(load_est.notes[0])

    # Speculative decode suggestion
    if result.model and result.model.model_id:
        draft = suggest_draft_model(result.model.model_id)
        if draft:
            runtime_notes.append(f"Speculative decode: use {draft['draft_model']} as draft ({draft['expected_speedup']} speedup)")

    if runtime_notes:
        result.explanation.append("")
        result.explanation.append("Runtime analysis:")
        result.explanation.extend(f"  {n}" for n in runtime_notes)
