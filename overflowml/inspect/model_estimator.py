"""Model size estimation — converts HF metadata into ModelInfo."""

from __future__ import annotations

import logging
from typing import Optional

from ..core.types import ModelInfo
from .arch_registry import classify_task, estimate_params_from_config
from .hf_probe import probe_config, probe_safetensors_size

logger = logging.getLogger("overflowml")


def inspect_model(
    model_id: str,
    trust_remote_code: bool = False,
) -> ModelInfo:
    """Inspect a HuggingFace model and estimate sizes at various dtypes.

    Tries safetensors index first (exact), then config.json estimation.
    Never downloads full weight files.
    """
    info = ModelInfo(model_id=model_id)

    # Strategy 1: safetensors index (exact byte count)
    total_bytes = probe_safetensors_size(model_id)
    if total_bytes is not None and total_bytes > 0:
        # safetensors stores in the native dtype (usually bf16/fp16 = 2 bytes/param)
        info.param_count = total_bytes // 2  # approximate
        info.source = "safetensors index"
        info.confidence = "high"
        info.notes.append(f"Size from safetensors metadata: {total_bytes / 1e9:.1f} GB")

    # Strategy 2: config.json
    config = probe_config(model_id, trust_remote_code)
    if config:
        architectures = config.get("architectures", [])
        if architectures:
            info.architecture = architectures[0]
        info.task_family = classify_task(
            info.architecture or "", model_id
        )

        if info.param_count is None:
            param_count, source = estimate_params_from_config(config)
            if param_count:
                info.param_count = param_count
                info.source = source
                info.confidence = "medium"
                info.notes.append(f"Params estimated from {source}")

    # Strategy 3: no data available
    if info.param_count is None:
        info.source = "unknown"
        info.confidence = "low"
        info.notes.append("Could not determine model size — specify manually with --assume-size-gb")
        return info

    # Compute size estimates at various dtypes
    params = info.param_count
    info.estimated_sizes_gb = {
        "fp32": params * 4 / (1024 ** 3),
        "fp16": params * 2 / (1024 ** 3),
        "int8": params * 1 / (1024 ** 3),
        "int4": params * 0.5 / (1024 ** 3),
    }

    return info


def estimate_size_gb(model_id: str, trust_remote_code: bool = False) -> float:
    """Quick helper — returns fp16 size in GB, or 14.0 as fallback."""
    info = inspect_model(model_id, trust_remote_code)
    if info.estimated_sizes_gb and "fp16" in info.estimated_sizes_gb:
        return max(info.estimated_sizes_gb["fp16"], 0.5)
    return 14.0
