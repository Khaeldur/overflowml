"""Runtime intelligence — KV cache, flash attention, PCIe, fragmentation.

These are the hidden factors that make "fits on paper" crash in practice.
The strategy engine handles weight placement. This module handles everything else.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("overflowml")


# ============================================================
# KV Cache Budgeting
# ============================================================

@dataclass
class KVCacheEstimate:
    """KV cache memory estimate for a given model + context length."""
    cache_gb: float = 0.0
    per_token_mb: float = 0.0
    context_length: int = 0
    num_layers: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    dtype_bytes: int = 2
    notes: list[str] = field(default_factory=list)


def estimate_kv_cache(
    num_layers: int = 32,
    num_kv_heads: int = 32,
    head_dim: int = 128,
    context_length: int = 4096,
    dtype_bytes: int = 2,
    batch_size: int = 1,
) -> KVCacheEstimate:
    """Estimate KV cache memory for a transformer model.

    KV cache = 2 (K+V) * num_layers * num_kv_heads * head_dim * context_length * dtype_bytes * batch_size
    """
    cache_bytes = 2 * num_layers * num_kv_heads * head_dim * context_length * dtype_bytes * batch_size
    cache_gb = cache_bytes / (1024 ** 3)
    per_token_bytes = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    per_token_mb = per_token_bytes / (1024 ** 2)

    est = KVCacheEstimate(
        cache_gb=cache_gb,
        per_token_mb=per_token_mb,
        context_length=context_length,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes,
    )

    if cache_gb > 1:
        est.notes.append(f"KV cache at {context_length} tokens: {cache_gb:.1f}GB ({per_token_mb:.1f}MB/token)")
    if batch_size > 1:
        est.notes.append(f"Batch size {batch_size} multiplies cache {batch_size}x")

    return est


def estimate_kv_cache_from_config(config: dict, context_length: int = 4096, batch_size: int = 1) -> KVCacheEstimate:
    """Estimate KV cache from a HuggingFace config dict."""
    hidden = config.get("hidden_size", 4096)
    layers = config.get("num_hidden_layers", 32)
    num_heads = config.get("num_attention_heads", 32)
    num_kv_heads = config.get("num_key_value_heads", num_heads)  # GQA support
    head_dim = config.get("head_dim", hidden // num_heads)
    max_ctx = config.get("max_position_embeddings", 4096)

    if context_length > max_ctx:
        context_length = max_ctx

    return estimate_kv_cache(
        num_layers=layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        context_length=context_length,
        batch_size=batch_size,
    )


# ============================================================
# Flash Attention Detection
# ============================================================

@dataclass
class FlashAttentionStatus:
    available: bool = False
    backend: str = "none"  # "flash_attn", "sdpa", "xformers", "none"
    version: str = ""
    notes: list[str] = field(default_factory=list)


def detect_flash_attention() -> FlashAttentionStatus:
    """Detect available efficient attention backends."""
    status = FlashAttentionStatus()

    # Check flash_attn package (fastest)
    try:
        import flash_attn
        status.available = True
        status.backend = "flash_attn"
        status.version = getattr(flash_attn, "__version__", "unknown")
        status.notes.append(f"flash_attn {status.version} available (fastest path)")
        return status
    except ImportError:
        pass

    # Check PyTorch SDPA (built-in since 2.0)
    try:
        import torch
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            status.available = True
            status.backend = "sdpa"
            status.version = torch.__version__

            # Check if SDPA has flash backend
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "flash_sdp_enabled"):
                if torch.backends.cuda.flash_sdp_enabled():
                    status.notes.append(f"PyTorch SDPA with flash backend (torch {status.version})")
                else:
                    status.notes.append(f"PyTorch SDPA available but flash backend disabled")
            else:
                status.notes.append(f"PyTorch SDPA available (torch {status.version})")
            return status
    except ImportError:
        pass

    # Check xformers
    try:
        import xformers
        status.available = True
        status.backend = "xformers"
        status.version = getattr(xformers, "__version__", "unknown")
        status.notes.append(f"xformers {status.version} available")
        return status
    except ImportError:
        pass

    status.notes.append("No efficient attention backend found — using naive attention (slow, high memory)")
    status.notes.append("Fix: pip install flash-attn (CUDA) or ensure torch >= 2.0 for SDPA")
    return status


# ============================================================
# PCIe Bandwidth Awareness
# ============================================================

@dataclass
class PCIeBandwidth:
    generation: int = 0    # 3, 4, 5
    width: int = 0         # x16, x8, etc.
    theoretical_gbps: float = 0.0
    practical_gbps: float = 0.0
    detected: bool = False
    notes: list[str] = field(default_factory=list)

PCIE_THEORETICAL = {
    (3, 16): 15.75,
    (4, 16): 31.5,
    (5, 16): 63.0,
    (3, 8): 7.88,
    (4, 8): 15.75,
    (5, 8): 31.5,
}


def detect_pcie_bandwidth() -> PCIeBandwidth:
    """Detect PCIe generation and estimate bandwidth."""
    bw = PCIeBandwidth()

    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=pcie.link.gen.current,pcie.link.width.current",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 2:
                bw.generation = int(parts[0].strip())
                bw.width = int(parts[1].strip())
                bw.detected = True
                bw.theoretical_gbps = PCIE_THEORETICAL.get((bw.generation, bw.width), 0)
                bw.practical_gbps = bw.theoretical_gbps * 0.7  # ~70% practical
                bw.notes.append(f"PCIe Gen{bw.generation} x{bw.width}: ~{bw.practical_gbps:.0f} GB/s practical")
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    if not bw.detected:
        bw.notes.append("PCIe bandwidth not detected (nvidia-smi unavailable)")

    return bw


def estimate_transfer_overhead(model_ram_gb: float, pcie: Optional[PCIeBandwidth] = None) -> dict:
    """Estimate time to transfer model data between GPU and RAM."""
    if pcie is None:
        pcie = detect_pcie_bandwidth()

    if pcie.practical_gbps <= 0:
        return {"transfer_time_s": 0, "note": "PCIe bandwidth unknown"}

    transfer_time = model_ram_gb / pcie.practical_gbps
    return {
        "ram_portion_gb": model_ram_gb,
        "bandwidth_gbps": pcie.practical_gbps,
        "transfer_time_s": transfer_time,
        "note": f"{model_ram_gb:.0f}GB over PCIe Gen{pcie.generation}: ~{transfer_time:.1f}s per full transfer",
    }


# ============================================================
# Fragmentation Diagnostics
# ============================================================

@dataclass
class FragmentationReport:
    allocated_gb: float = 0.0
    reserved_gb: float = 0.0
    fragmentation_ratio: float = 0.0  # (reserved - allocated) / reserved
    is_fragmented: bool = False
    notes: list[str] = field(default_factory=list)


def diagnose_fragmentation() -> FragmentationReport:
    """Check CUDA memory for fragmentation."""
    report = FragmentationReport()

    try:
        import torch
        if not torch.cuda.is_available():
            report.notes.append("No CUDA device — fragmentation check skipped")
            return report

        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        report.allocated_gb = allocated
        report.reserved_gb = reserved

        if reserved > 0:
            report.fragmentation_ratio = (reserved - allocated) / reserved
            if report.fragmentation_ratio > 0.3:
                report.is_fragmented = True
                waste = reserved - allocated
                report.notes.append(
                    f"VRAM fragmentation: {waste:.1f}GB wasted "
                    f"({report.fragmentation_ratio:.0%} of reserved memory is unused holes)"
                )
                report.notes.append("Fix: gc.collect() + torch.cuda.empty_cache(), or restart process")
            else:
                report.notes.append(f"VRAM healthy: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
        else:
            report.notes.append("No VRAM allocated yet")

    except ImportError:
        report.notes.append("torch not available")

    return report


# ============================================================
# Per-Layer Precision Planning
# ============================================================

@dataclass
class LayerPrecisionPlan:
    """Per-layer quantization recommendation."""
    layers: list[dict] = field(default_factory=list)
    total_original_gb: float = 0.0
    total_optimized_gb: float = 0.0
    savings_pct: float = 0.0
    notes: list[str] = field(default_factory=list)


def plan_layer_precision(
    num_layers: int = 32,
    hidden_size: int = 4096,
    vocab_size: int = 32000,
    intermediate_size: int = 16384,
    model_size_gb: float = 0.0,
) -> LayerPrecisionPlan:
    """Plan per-layer quantization for optimal quality/size tradeoff.

    Principle: not all layers are equal.
    - Embedding + LM head: keep FP16 (high quality impact)
    - First/last 2 attention layers: keep FP16 (high quality impact)
    - Middle attention layers: INT8 safe
    - FFN layers: INT4 safe (most parameters, least quality sensitive)
    """
    plan = LayerPrecisionPlan()

    # Embedding layer
    embed_params = vocab_size * hidden_size
    embed_gb_fp16 = embed_params * 2 / (1024 ** 3)
    plan.layers.append({
        "name": "embedding", "precision": "fp16",
        "params": embed_params, "size_gb": embed_gb_fp16,
        "reason": "embeddings are quality-critical — keep full precision",
    })

    # Transformer layers
    attn_params_per_layer = 4 * hidden_size * hidden_size
    ffn_params_per_layer = 3 * hidden_size * intermediate_size

    for i in range(num_layers):
        # First/last 2 layers: FP16 (quality-critical)
        if i < 2 or i >= num_layers - 2:
            attn_prec = "fp16"
            ffn_prec = "fp16"
            attn_reason = "first/last layers are quality-critical"
            ffn_reason = "first/last layers are quality-critical"
        # Middle layers: INT8 attention, INT4 FFN
        else:
            attn_prec = "int8"
            ffn_prec = "int4"
            attn_reason = "middle attention layers tolerate INT8 well"
            ffn_reason = "FFN layers have most params but least quality sensitivity"

        attn_bytes = {"fp16": 2, "int8": 1, "int4": 0.5}[attn_prec]
        ffn_bytes = {"fp16": 2, "int8": 1, "int4": 0.5}[ffn_prec]

        plan.layers.append({
            "name": f"layer_{i}_attention", "precision": attn_prec,
            "params": attn_params_per_layer,
            "size_gb": attn_params_per_layer * attn_bytes / (1024 ** 3),
            "reason": attn_reason,
        })
        plan.layers.append({
            "name": f"layer_{i}_ffn", "precision": ffn_prec,
            "params": ffn_params_per_layer,
            "size_gb": ffn_params_per_layer * ffn_bytes / (1024 ** 3),
            "reason": ffn_reason,
        })

    # LM head
    head_gb_fp16 = embed_gb_fp16  # typically tied weights
    plan.layers.append({
        "name": "lm_head", "precision": "fp16",
        "params": embed_params, "size_gb": head_gb_fp16,
        "reason": "output projection is quality-critical",
    })

    plan.total_optimized_gb = sum(l["size_gb"] for l in plan.layers)
    plan.total_original_gb = model_size_gb if model_size_gb > 0 else sum(
        l["params"] * 2 / (1024 ** 3) for l in plan.layers
    )

    if plan.total_original_gb > 0:
        plan.savings_pct = (1 - plan.total_optimized_gb / plan.total_original_gb) * 100
        plan.notes.append(
            f"Mixed precision: {plan.total_original_gb:.1f}GB -> {plan.total_optimized_gb:.1f}GB "
            f"({plan.savings_pct:.0f}% smaller)"
        )
        plan.notes.append("Critical layers (embed, first/last 2, lm_head): FP16")
        plan.notes.append("Middle attention: INT8, Middle FFN: INT4")

    return plan


# ============================================================
# Context-Length-Aware Strategy Switching
# ============================================================

def context_adjusted_vram(
    model_weights_gb: float,
    context_length: int,
    num_layers: int = 32,
    num_kv_heads: int = 32,
    head_dim: int = 128,
    batch_size: int = 1,
) -> dict:
    """Calculate total VRAM needed: weights + KV cache + activations.

    This is what the planner SHOULD use instead of just weight size.
    """
    kv = estimate_kv_cache(num_layers, num_kv_heads, head_dim, context_length, batch_size=batch_size)

    # Activation memory: ~2x one layer's activations during forward pass
    activation_gb = (2 * num_kv_heads * head_dim * context_length * 4) / (1024 ** 3)  # float32 activations

    total = model_weights_gb + kv.cache_gb + activation_gb

    return {
        "weights_gb": model_weights_gb,
        "kv_cache_gb": kv.cache_gb,
        "activation_gb": activation_gb,
        "total_gb": total,
        "context_length": context_length,
        "breakdown": f"Weights {model_weights_gb:.1f}GB + KV cache {kv.cache_gb:.1f}GB + Activations {activation_gb:.1f}GB = {total:.1f}GB",
    }


# ============================================================
# Startup / Load Time Hints
# ============================================================

@dataclass
class LoadTimeEstimate:
    estimated_seconds: float = 0.0
    model_size_gb: float = 0.0
    storage_type: str = "unknown"  # "ssd", "hdd", "network"
    notes: list[str] = field(default_factory=list)


def estimate_load_time(model_size_gb: float, use_mmap: bool = False) -> LoadTimeEstimate:
    """Estimate model loading time."""
    est = LoadTimeEstimate(model_size_gb=model_size_gb)

    # Assume SSD (~2 GB/s read), HDD (~0.15 GB/s), network (~0.1 GB/s)
    ssd_speed = 2.0 if not use_mmap else 0.1  # mmap is lazy, near-instant start
    est.storage_type = "ssd"
    est.estimated_seconds = model_size_gb / ssd_speed

    if use_mmap:
        est.notes.append(f"Memory-mapped loading: near-instant start, pages loaded on demand")
        est.estimated_seconds = 2.0  # ~2s for mmap setup regardless of size
    elif model_size_gb > 50:
        est.notes.append(f"Large model ({model_size_gb:.0f}GB): ~{est.estimated_seconds:.0f}s load time from SSD")
        est.notes.append("Tip: use safetensors format for fastest loading")
    else:
        est.notes.append(f"Estimated load time: ~{est.estimated_seconds:.0f}s from SSD")

    if model_size_gb > 100:
        est.notes.append("Consider: safetensors with mmap=True for instant startup")

    return est


# ============================================================
# Speculative Decode Pairing
# ============================================================

DRAFT_MODEL_PAIRS = {
    # Large model -> suggested draft model
    "llama-3-70b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama-3.1-70b": "meta-llama/Llama-3.2-1B",
    "llama-3.1-405b": "meta-llama/Llama-3.2-3B",
    "qwen": "Qwen/Qwen2.5-0.5B",
    "mistral-7b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mixtral": "mistralai/Mistral-7B-v0.1",
    "phi-3": "microsoft/phi-1_5",
    "gemma-2": "google/gemma-2-2b",
    "command-r": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


def suggest_draft_model(model_id: str) -> Optional[dict]:
    """Suggest a draft model for speculative decoding."""
    model_lower = model_id.lower()
    for pattern, draft in DRAFT_MODEL_PAIRS.items():
        if pattern in model_lower:
            return {
                "draft_model": draft,
                "target_model": model_id,
                "expected_speedup": "2-3x",
                "note": f"Use {draft} as draft model for speculative decoding. "
                        f"Small model drafts tokens, {model_id} verifies in batch. "
                        f"No quality loss — rejected drafts are resampled.",
            }
    return None


# ============================================================
# Prefix Cache Awareness
# ============================================================

def estimate_prefix_savings(
    num_requests: int,
    shared_prefix_tokens: int,
    total_tokens_per_request: int,
    num_layers: int = 32,
    num_kv_heads: int = 32,
    head_dim: int = 128,
) -> dict:
    """Estimate memory savings from prefix caching."""
    per_token_bytes = 2 * num_layers * num_kv_heads * head_dim * 2  # K+V, fp16

    # Without prefix caching: each request stores full KV
    without_cache = num_requests * total_tokens_per_request * per_token_bytes
    # With prefix caching: shared prefix stored once
    with_cache = (shared_prefix_tokens * per_token_bytes +
                  num_requests * (total_tokens_per_request - shared_prefix_tokens) * per_token_bytes)

    savings_gb = (without_cache - with_cache) / (1024 ** 3)
    savings_pct = (1 - with_cache / without_cache) * 100 if without_cache > 0 else 0

    return {
        "without_cache_gb": without_cache / (1024 ** 3),
        "with_cache_gb": with_cache / (1024 ** 3),
        "savings_gb": savings_gb,
        "savings_pct": savings_pct,
        "note": f"Prefix caching saves {savings_gb:.1f}GB ({savings_pct:.0f}%) "
                f"by sharing {shared_prefix_tokens} tokens across {num_requests} requests",
    }
