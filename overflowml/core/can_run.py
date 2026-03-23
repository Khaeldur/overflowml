"""CI/CD gating API — check if a model can run on this hardware."""

from __future__ import annotations

from typing import Optional, Union

from .types import CanRunResult, HardwareInfo
from .hardware import detect_hardware_info, hardware_info_to_legacy


_OFFLOAD_SEVERITY = {
    "none": 0,
    "model_cpu": 1,
    "expert_offload": 1,
    "layer_hybrid": 2,
    "sequential_cpu": 3,
    "disk": 4,
}


def can_run(
    model_or_size: Union[str, float],
    hw: Optional[HardwareInfo] = None,
    *,
    max_offload: str = "sequential_cpu",
    trust_remote_code: bool = False,
) -> CanRunResult:
    """Check if this hardware can run a model of the given size.

    Args:
        model_or_size: HuggingFace model ID or size in GB.
        hw: Hardware info. Auto-detected if None.
        max_offload: Maximum acceptable offload mode.
            "none" = must fit in VRAM, "model_cpu" = CPU offload OK,
            "sequential_cpu" = sequential OK (default), "disk" = anything goes.
        trust_remote_code: Passed to model inspection.

    Returns:
        CanRunResult with ok, reason, and hardware info.
    """
    if hw is None:
        hw = detect_hardware_info()

    # Resolve model size
    if isinstance(model_or_size, str):
        try:
            model_size_gb = float(model_or_size)
        except ValueError:
            try:
                from ..inspect import estimate_size_gb
                model_size_gb = estimate_size_gb(model_or_size, trust_remote_code)
            except Exception as e:
                return CanRunResult(
                    ok=False,
                    reason=f"Could not estimate size for '{model_or_size}': {e}",
                    detected_vram_gb=hw.total_vram_gb,
                    detected_ram_gb=hw.total_ram_gb,
                )
    else:
        model_size_gb = float(model_or_size)

    # Get strategy via legacy path
    legacy_hw = hardware_info_to_legacy(hw)
    from ..strategy import pick_strategy
    s = pick_strategy(legacy_hw, model_size_gb)

    # Check offload severity
    strategy_severity = _OFFLOAD_SEVERITY.get(s.offload.value, 3)
    max_severity = _OFFLOAD_SEVERITY.get(max_offload, 2)

    if s.offload.value == "disk":
        return CanRunResult(
            ok=False,
            reason=f"Insufficient memory — requires disk offload (model: {model_size_gb:.0f}GB, VRAM: {hw.total_vram_gb:.0f}GB, RAM: {hw.total_ram_gb:.0f}GB)",
            detected_vram_gb=hw.total_vram_gb,
            detected_ram_gb=hw.total_ram_gb,
        )

    # Even with INT4+sequential, check if quantized model exceeds RAM
    int4_size = model_size_gb * 0.3
    if int4_size > hw.total_ram_gb and s.offload.value == "sequential_cpu":
        return CanRunResult(
            ok=False,
            reason=f"Model too large even with INT4 ({int4_size:.0f}GB) — exceeds RAM ({hw.total_ram_gb:.0f}GB)",
            detected_vram_gb=hw.total_vram_gb,
            detected_ram_gb=hw.total_ram_gb,
        )

    if strategy_severity > max_severity:
        return CanRunResult(
            ok=False,
            reason=f"Requires {s.offload.value} but max allowed is {max_offload}",
            recommended_strategy=s.offload.value,
            detected_vram_gb=hw.total_vram_gb,
            detected_ram_gb=hw.total_ram_gb,
        )

    # Build strategy name
    parts = []
    if s.quantization.value != "none":
        parts.append(s.quantization.value.upper())
    if s.offload.value != "none":
        parts.append(s.offload.value)
    strategy_name = " + ".join(parts) if parts else "direct load"

    reason = s.notes[0] if s.notes else f"Model can run via {strategy_name}"

    return CanRunResult(
        ok=True,
        reason=reason,
        recommended_strategy=strategy_name,
        detected_vram_gb=hw.total_vram_gb,
        detected_ram_gb=hw.total_ram_gb,
    )
