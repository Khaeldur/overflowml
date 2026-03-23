"""Auto-batching engine — calculates optimal batch size from VRAM headroom.

This is OverflowML's second core capability:
1. Strategy engine: "I know how to LOAD this model"
2. Auto-batching: "I know how much you can PUSH through it"

Users either set batch_size=1 (safe but slow) or guess too high and crash.
OverflowML measures remaining VRAM after model loading and calculates the
optimal batch size that fills available memory without OOM.
"""

from __future__ import annotations

import gc
import logging
import math
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence

logger = logging.getLogger("overflowml")


@dataclass
class BatchConfig:
    """Result of auto-batch calculation."""
    batch_size: int = 1
    available_vram_gb: float = 0.0
    estimated_per_item_gb: float = 0.0
    safety_margin: float = 0.85
    method: str = "unknown"
    notes: list[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


def measure_vram_headroom() -> tuple[float, float]:
    """Measure available VRAM after model loading.

    Returns (available_gb, total_gb). Available = total - currently allocated.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            available = total - allocated
            return max(0, available), total
    except Exception:
        pass

    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            import psutil
            mem = psutil.virtual_memory()
            total = mem.total / (1024 ** 3)
            available = mem.available / (1024 ** 3)
            return available * 0.75, total  # MPS shares RAM, be conservative
    except Exception:
        pass

    # CPU fallback — use RAM
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.available / (1024 ** 3), mem.total / (1024 ** 3)
    except ImportError:
        return 0.0, 0.0


def estimate_per_item_memory(
    pipe_or_model: Any,
    sample_input: Any = None,
    input_shape: Optional[tuple] = None,
) -> float:
    """Estimate memory cost per batch item in GB.

    Strategy: run a single forward pass, measure the VRAM spike, that's the
    per-item cost. If we can't run a forward pass, estimate from model arch.
    """
    # Method 1: measure actual forward pass
    if sample_input is not None:
        measured = _measure_forward_pass(pipe_or_model, sample_input)
        if measured > 0:
            return measured

    # Method 2: estimate from model architecture
    return _estimate_from_arch(pipe_or_model, input_shape)


def calculate_batch_size(
    pipe_or_model: Any = None,
    per_item_gb: Optional[float] = None,
    available_vram_gb: Optional[float] = None,
    safety_margin: float = 0.85,
    max_batch_size: int = 64,
    min_batch_size: int = 1,
) -> BatchConfig:
    """Calculate optimal batch size from VRAM headroom.

    Args:
        pipe_or_model: The loaded model/pipeline (for arch-based estimation).
        per_item_gb: Known per-item memory cost. Auto-estimated if None.
        available_vram_gb: Available VRAM. Auto-measured if None.
        safety_margin: Use this fraction of available VRAM (default 85%).
        max_batch_size: Upper bound on batch size.
        min_batch_size: Lower bound (default 1).

    Returns:
        BatchConfig with optimal batch_size and metadata.
    """
    config = BatchConfig(safety_margin=safety_margin)

    # Measure headroom
    if available_vram_gb is None:
        available, total = measure_vram_headroom()
        config.available_vram_gb = available
        config.notes.append(f"VRAM headroom: {available:.1f}GB free of {total:.0f}GB total")
    else:
        config.available_vram_gb = available_vram_gb

    usable = config.available_vram_gb * safety_margin

    if usable <= 0:
        config.batch_size = min_batch_size
        config.method = "no_vram"
        config.notes.append("No VRAM headroom — batch_size=1")
        return config

    # Estimate per-item cost
    if per_item_gb is None and pipe_or_model is not None:
        per_item_gb = _estimate_from_arch(pipe_or_model)
    if per_item_gb is None or per_item_gb <= 0:
        per_item_gb = 0.5  # conservative default
        config.method = "default_estimate"
        config.notes.append("Could not estimate per-item cost — using 0.5GB default")
    else:
        config.method = "architecture_estimate"

    config.estimated_per_item_gb = per_item_gb

    # Calculate
    optimal = int(usable / per_item_gb)
    optimal = max(min_batch_size, min(optimal, max_batch_size))

    config.batch_size = optimal
    config.notes.append(f"Optimal batch_size={optimal} ({per_item_gb:.2f}GB/item x {optimal} = {per_item_gb * optimal:.1f}GB of {usable:.1f}GB usable)")

    return config


def auto_batch(
    items: Sequence,
    pipe_or_model: Any = None,
    batch_size: Optional[int] = None,
    per_item_gb: Optional[float] = None,
    safety_margin: float = 0.85,
    max_batch_size: int = 64,
    cleanup_between: bool = True,
) -> Iterator[list]:
    """Yield optimally-sized batches from a sequence of items.

    Usage:
        for batch in overflowml.auto_batch(prompts, pipe):
            results = pipe(batch)

    If batch_size is None, automatically calculates from VRAM headroom.
    Cleans up VRAM between batches to prevent accumulation.
    """
    if batch_size is None:
        config = calculate_batch_size(
            pipe_or_model=pipe_or_model,
            per_item_gb=per_item_gb,
            safety_margin=safety_margin,
            max_batch_size=max_batch_size,
        )
        batch_size = config.batch_size
        logger.info("Auto-batch: %s", "; ".join(config.notes))

    items_list = list(items)
    for i in range(0, len(items_list), batch_size):
        if cleanup_between and i > 0:
            _cleanup_vram()
        yield items_list[i:i + batch_size]


# --- Internal helpers ---

def _measure_forward_pass(pipe_or_model: Any, sample_input: Any) -> float:
    """Run a single forward pass and measure VRAM spike."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()

        with torch.inference_mode():
            if callable(pipe_or_model):
                pipe_or_model(sample_input)
            elif hasattr(pipe_or_model, "generate"):
                pipe_or_model.generate(sample_input, max_new_tokens=1)

        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        spike_gb = (peak - before) / (1024 ** 3)
        return max(spike_gb, 0.01)
    except Exception as e:
        logger.debug("Forward pass measurement failed: %s", e)
        return 0.0


def _estimate_from_arch(pipe_or_model: Any, input_shape: Optional[tuple] = None) -> float:
    """Estimate per-item memory from model architecture."""
    # Diffusers pipeline: estimate from image resolution
    if hasattr(pipe_or_model, "unet") or hasattr(pipe_or_model, "transformer"):
        # Typical diffusion: ~2-4GB per image at 1024x1024
        # Scale by resolution if known
        return 3.0

    # Transformer/LLM: estimate from hidden size and sequence length
    try:
        config = getattr(pipe_or_model, "config", None)
        if config:
            hidden = getattr(config, "hidden_size", 4096)
            seq_len = getattr(config, "max_position_embeddings", 2048)
            if input_shape:
                seq_len = input_shape[-1] if len(input_shape) > 1 else input_shape[0]
            # KV cache + activations: ~2 bytes per element per layer
            layers = getattr(config, "num_hidden_layers", 32)
            # Rough: 2 * hidden * seq_len * layers * 2 bytes (KV) + activations
            kv_bytes = 2 * hidden * seq_len * layers * 2
            activation_bytes = hidden * seq_len * 4  # float32 activations
            total_gb = (kv_bytes + activation_bytes) / (1024 ** 3)
            return max(total_gb, 0.1)
    except Exception:
        pass

    return 0.5  # conservative fallback


def _cleanup_vram():
    """Clean up VRAM between batches."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
    except Exception:
        pass
