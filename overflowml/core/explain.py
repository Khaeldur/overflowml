"""Explanation layer — builds human-readable reasoning for strategy decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import HardwareInfo, ModelInfo, PlanResult, StrategyCandidate


def build_explanation(
    model_size_gb: float,
    hw_info: "HardwareInfo",
    recommended: "StrategyCandidate | None",
    all_strategies: list["StrategyCandidate"],
    model_info: "ModelInfo | None" = None,
) -> list[str]:
    """Build a decision-chain explanation for why a strategy was selected.

    Every explanation must reference at least one trap that was avoided.
    This is OverflowML's core identity: gotcha-aware strategy planning.
    """
    lines = []

    # Model info
    if model_info and model_info.param_count:
        params_b = model_info.param_count / 1e9
        lines.append(f"Model estimated at {params_b:.1f}B params ({model_info.source})")
    lines.append(f"fp16 weight footprint: {model_size_gb:.1f} GB")

    # Hardware
    if hw_info.gpus:
        gpu = hw_info.gpus[0]
        if hw_info.num_gpus > 1:
            lines.append(f"Detected {hw_info.num_gpus}x {gpu.name} ({gpu.total_vram_gb:.0f}GB each, {hw_info.total_vram_gb:.0f}GB total)")
        else:
            lines.append(f"Detected GPU: {gpu.name} ({gpu.total_vram_gb:.0f}GB VRAM)")
    else:
        lines.append("No GPU detected — CPU only")
    lines.append(f"System RAM: {hw_info.total_ram_gb:.0f}GB")

    # What was rejected and why
    rejected = [s for s in all_strategies if not s.viable]
    for s in rejected:
        lines.append(f"Rejected: {s.name} — {s.rejection_reason}")

    # What was chosen and why
    if recommended:
        lines.append(f"Selected: {recommended.name}")
        for note in recommended.notes:
            lines.append(f"  {note}")

    # Always reference known traps that were considered
    trap_lines = _get_trap_context(hw_info, recommended, all_strategies)
    if trap_lines:
        lines.append("")
        lines.append("Known traps handled:")
        lines.extend(trap_lines)

    return lines


def _get_trap_context(
    hw_info: "HardwareInfo",
    recommended: "StrategyCandidate | None",
    all_strategies: list["StrategyCandidate"],
) -> list[str]:
    """Generate trap-specific context lines based on hardware and strategy."""
    traps = []
    platform = hw_info.platform.split("-")[0] if hw_info.platform else ""

    # FP8 + offload on Windows
    has_offload = any("cpu" in (s.name or "").lower() for s in all_strategies if s.viable)
    if platform == "Windows" and has_offload:
        traps.append("  - FP8 + CPU offload crashes on Windows (Float8Tensor can't move between devices) — FP8 only used without offload")

    # attention_slicing + sequential
    has_sequential = any("sequential" in (s.name or "").lower() for s in all_strategies if s.viable)
    if has_sequential:
        traps.append("  - attention_slicing disabled with sequential offload (causes CUDA illegal memory access)")

    # expandable_segments on Windows
    if platform == "Windows":
        traps.append("  - expandable_segments disabled (not supported on Windows WDDM)")

    # FP8 without torchao
    if not hw_info.supports_fp8:
        has_fp8_candidate = any("FP8" in s.name for s in all_strategies)
        if has_fp8_candidate:
            traps.append("  - FP8 quantization unavailable (torchao not installed)")

    return traps
