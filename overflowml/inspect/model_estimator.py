"""Model size estimation — converts HF metadata into ModelInfo."""

from __future__ import annotations

import logging
import os

from ..core.types import ModelInfo
from .arch_registry import classify_task, estimate_params_from_config
from .hf_probe import ModelSizeUnknown, probe_config, probe_safetensors_params, probe_stored_params

logger = logging.getLogger("overflowml")

GIB = 1024 ** 3

# Bits per element for safetensors dtypes; unlisted dtypes are counted as 8 bits
DTYPE_BITS = {
    "F64": 64, "I64": 64, "U64": 64,
    "F32": 32, "I32": 32, "U32": 32,
    "F16": 16, "BF16": 16, "I16": 16, "U16": 16,
    "F8_E4M3": 8, "F8_E5M2": 8, "F8_E8M0": 8, "I8": 8, "U8": 8, "BOOL": 8,
    "F6_E2M3": 6, "F6_E3M2": 6, "F4": 4,
}
FLOAT_DTYPES = {"F64", "F32", "F16", "BF16"}


def weights_bytes(params_by_dtype: dict[str, int]) -> int:
    return sum(n * DTYPE_BITS.get(dtype, 8) // 8 for dtype, n in params_by_dtype.items())


def is_prequantized(params_by_dtype: dict[str, int]) -> bool:
    """True when packed/quantized tensors (U8, I32, F8, ...) hold >5% of the bytes.

    A small share is normal: int64 position_ids buffers ship in plain fp16 checkpoints.
    """
    total = weights_bytes(params_by_dtype)
    quantized = weights_bytes({d: n for d, n in params_by_dtype.items() if d not in FLOAT_DTYPES})
    return total > 0 and quantized > 0.05 * total


def inspect_model(
    model_id: str,
    trust_remote_code: bool = False,
    use_cache: bool = True,
) -> ModelInfo:
    """Inspect a HuggingFace model and estimate sizes at various dtypes.

    Tries safetensors metadata first (exact), then config.json estimation.
    Never downloads weight files.

    Raises ModelSizeUnknown if the repo doesn't exist, is gated, or the Hub
    is unreachable with nothing cached. A reachable repo without metadata
    returns a low-confidence ModelInfo with no sizes.
    """
    # Local directories are cheap to re-read and their contents change under the same path
    use_cache = use_cache and not os.path.isdir(model_id)
    if use_cache:
        try:
            from ..core.cache import load_cached_model
            import dataclasses
            cached = load_cached_model(model_id)
            if cached:
                cached.pop("_version", None)
                cached.pop("_timestamp", None)
                cached.pop("_schema", None)
                return ModelInfo(**{k: v for k, v in cached.items() if k in {f.name for f in dataclasses.fields(ModelInfo)}})
        except Exception:
            pass

    info = ModelInfo(model_id=model_id)

    # Strategy 1: safetensors metadata (exact per-dtype parameter counts)
    hub_error = None
    try:
        probe = probe_safetensors_params(model_id)
    except ModelSizeUnknown as e:
        if e.reason != "offline":
            raise
        hub_error = e  # the config may still be in the local HF cache
        probe = None

    if probe:
        params_by_dtype = probe.params_by_dtype
        info.source = "safetensors metadata"
        info.confidence = "high"
        if is_prequantized(params_by_dtype):
            # Hub counts packed weights as unpacked params; size from the stored headers
            stored = probe_stored_params(model_id)
            if stored:
                size_bytes = weights_bytes(stored)
            else:
                size_bytes = weights_bytes(params_by_dtype)
                info.confidence = "low"
                info.notes.append("Could not read safetensors headers; packed weights may be overstated")
            info.prequantized = True
            info.estimated_sizes_gb = {"native": size_bytes / GIB}
            dtypes = ", ".join(sorted(params_by_dtype))
            info.notes.append(
                f"Pre-quantized checkpoint ({dtypes}): size is the stored weights, "
                "quantizing again won't shrink it"
            )
        else:
            size_bytes = weights_bytes(params_by_dtype)
            info.param_count = sum(params_by_dtype.values())
        info.notes.append(f"Size from safetensors metadata: {size_bytes / GIB:.1f} GiB as stored")
        if probe.pipeline:
            info.task_family = "diffusers"
            info.confidence = "medium"
            info.notes.append(
                "Diffusers pipeline: only the main component is counted — "
                "text encoders and VAE add more"
            )

    # Strategy 2: config.json
    config = probe_config(model_id, trust_remote_code)
    if config:
        architectures = config.get("architectures", [])
        if architectures:
            info.architecture = architectures[0]
        if info.task_family == "unknown":
            info.task_family = classify_task(info.architecture or "", model_id)

        if info.param_count is None and not info.estimated_sizes_gb:
            if config.get("quantization_config"):
                # An fp16 estimate of a quantized checkpoint would be several times too large
                if hub_error:
                    raise hub_error
                raise ModelSizeUnknown(model_id, "no_metadata", "quantized checkpoint without safetensors metadata")
            param_count, source = estimate_params_from_config(config)
            if param_count:
                info.param_count = param_count
                info.source = source
                info.confidence = "medium"
                info.notes.append(f"Params estimated from {source}")

    if hub_error and info.param_count is None and not info.estimated_sizes_gb:
        raise hub_error

    # Strategy 3: no data available
    if info.param_count is None and not info.estimated_sizes_gb:
        info.source = "unknown"
        info.confidence = "low"
        info.notes.append("Could not determine model size — pass the size in GB instead")
        return info

    # Compute size estimates at various dtypes
    if info.param_count is not None:
        params = info.param_count
        info.estimated_sizes_gb = {
            "fp32": params * 4 / GIB,
            "fp16": params * 2 / GIB,
            "int8": params * 1 / GIB,
            "int4": params * 0.5 / GIB,
        }

    # Save to cache — never a degraded offline estimate, which would outlive the outage
    if use_cache and not hub_error:
        try:
            import dataclasses
            from ..core.cache import save_cached_model
            save_cached_model(model_id, dataclasses.asdict(info))
        except Exception:
            pass

    return info


def size_label(info: ModelInfo) -> str:
    """How planning_size_gb's number should be described to users."""
    return "fp16" if "fp16" in (info.estimated_sizes_gb or {}) else "as stored"


def planning_size_gb(info: ModelInfo) -> float:
    """Size to plan with: fp16 weights, or stored bytes for pre-quantized checkpoints.

    Raises ModelSizeUnknown when inspection found no size.
    """
    sizes = info.estimated_sizes_gb or {}
    if "fp16" in sizes:
        return max(sizes["fp16"], 0.5)
    if "native" in sizes:
        return max(sizes["native"], 0.5)
    raise ModelSizeUnknown(info.model_id, "no_metadata")


def estimate_size_gb(model_id: str, trust_remote_code: bool = False) -> float:
    """Size in GB to plan with (see planning_size_gb). Raises ModelSizeUnknown."""
    return planning_size_gb(inspect_model(model_id, trust_remote_code))
