"""Apply optimization strategy to models and pipelines.

DEPRECATED: Direct imports from overflowml.optimize will be removed in v1.0.
Use overflowml.optimize_pipeline and overflowml.optimize_model from the top-level package.
"""

from __future__ import annotations

import warnings as _warnings
_warnings.warn(
    "Direct import from overflowml.optimize is deprecated. "
    "Use 'from overflowml import optimize_pipeline' instead. Will be removed in v1.0.",
    DeprecationWarning,
    stacklevel=2,
)

import gc
import logging
from typing import Any, Optional

from .detect import Accelerator, HardwareProfile, detect_hardware
from .strategy import DistributionMode, OffloadMode, QuantMode, Strategy, pick_strategy

logger = logging.getLogger("overflowml")


def optimize_pipeline(
    pipe: Any,
    model_size_gb: Optional[float] = None,
    hw: Optional[HardwareProfile] = None,
    strategy: Optional[Strategy] = None,
    *,
    prefer_speed: bool = True,
    allow_quantization: bool = True,
    compile: Optional[bool] = None,
    verbose: bool = True,
    lora_size_gb: Optional[float] = None,
) -> Strategy:
    """Optimize a diffusers pipeline for this hardware.

    Usage:
        from diffusers import StableDiffusionPipeline
        import overflowml

        pipe = StableDiffusionPipeline.from_pretrained("model", torch_dtype=torch.bfloat16)
        strategy = overflowml.optimize_pipeline(pipe, model_size_gb=40)
        # pipe is now optimized in-place — just call pipe(...)

    Args:
        pipe: A diffusers pipeline (any type)
        model_size_gb: Total model size in GB. Auto-estimated if not provided.
        hw: Hardware profile. Auto-detected if not provided.
        strategy: Pre-computed strategy. Auto-computed if not provided.
        prefer_speed: Prefer faster strategies over lower VRAM usage.
        allow_quantization: Allow quantization (FP8/INT8/INT4).
        compile: Override torch.compile (None = auto from strategy).
        verbose: Print what's happening.

    Returns:
        The Strategy that was applied.
    """
    if not _is_diffusers_pipeline(pipe):
        logger.warning(
            "Object doesn't look like a diffusers pipeline (missing offload methods). "
            "Use optimize_model() for standalone nn.Module models instead."
        )
        return optimize_model(pipe, model_size_gb=model_size_gb, hw=hw, strategy=strategy, verbose=verbose)

    if hw is None:
        hw = detect_hardware()

    if model_size_gb is None:
        model_size_gb = _estimate_model_size(pipe)

    if lora_size_gb and lora_size_gb > 0:
        model_size_gb += lora_size_gb
        if verbose:
            logger.info("LoRA adapter: +%.1fGB → effective model size %.1fGB", lora_size_gb, model_size_gb)

    if strategy is None:
        strategy = pick_strategy(
            hw, model_size_gb,
            prefer_speed=prefer_speed,
            allow_quantization=allow_quantization,
        )

    if verbose:
        logger.info("OverflowML: Hardware detected\n%s", hw.summary())
        logger.info("OverflowML: Strategy selected\n%s", strategy.summary())

    _apply_strategy(pipe, strategy, hw, compile_override=compile, verbose=verbose)

    # Auto-calculate optimal batch size from remaining VRAM
    try:
        from .batch import calculate_batch_size
        batch_config = calculate_batch_size(pipe_or_model=pipe)
        strategy.notes.append(f"Optimal batch_size={batch_config.batch_size} ({batch_config.estimated_per_item_gb:.1f}GB/item, {batch_config.available_vram_gb:.1f}GB headroom)")
        if verbose:
            logger.info("Auto-batch: optimal batch_size=%d", batch_config.batch_size)
    except Exception:
        pass

    return strategy


def optimize_model(
    model: Any,
    model_size_gb: Optional[float] = None,
    hw: Optional[HardwareProfile] = None,
    strategy: Optional[Strategy] = None,
    *,
    verbose: bool = True,
) -> Strategy:
    """Optimize a standalone model (transformers, any nn.Module).

    For models that aren't diffusers pipelines. Applies quantization and
    compilation but not pipeline-specific optimizations like attention slicing.
    """
    import torch

    if hw is None:
        hw = detect_hardware()

    if model_size_gb is None:
        model_size_gb = _count_params_gb(model)

    if strategy is None:
        strategy = pick_strategy(hw, model_size_gb)

    if verbose:
        logger.info("OverflowML: Optimizing model (%s)", type(model).__name__)
        logger.info("Strategy:\n%s", strategy.summary())

    # Quantization
    if strategy.quantization == QuantMode.FP8 and hw.supports_fp8:
        if strategy.offload == OffloadMode.NONE:
            try:
                from torchao.quantization import quantize_
                fp8_cfg = _load_fp8_config()
                if fp8_cfg is not None:
                    quantize_(model, fp8_cfg)
                    if verbose:
                        logger.info("Applied FP8 weight-only quantization")
                else:
                    logger.warning("FP8 quantization skipped: no supported Float8WeightOnly config found in torchao")
            except Exception as e:
                logger.warning("FP8 quantization failed: %s", e)

    # Device placement
    if strategy.offload == OffloadMode.NONE:
        device = _pick_device(hw)
        model.to(device)

    # Compile
    if strategy.compile:
        try:
            model = torch.compile(model, mode=strategy.compile_mode, fullgraph=False)
            if verbose:
                logger.info("Applied torch.compile (mode=%s)", strategy.compile_mode)
        except Exception as e:
            logger.warning("torch.compile failed: %s", e)

    return strategy


def _apply_strategy(pipe: Any, strategy: Strategy, hw: HardwareProfile,
                    compile_override: Optional[bool] = None, verbose: bool = True):
    """Apply a strategy to a diffusers pipeline."""
    import torch

    # --- Quantization (must happen before offload)
    if strategy.quantization == QuantMode.FP8 and strategy.offload == OffloadMode.NONE:
        if hw.supports_fp8:
            try:
                from torchao.quantization import quantize_
                fp8_cfg = _load_fp8_config()
                if fp8_cfg is not None:
                    if hasattr(pipe, "transformer"):
                        quantize_(pipe.transformer, fp8_cfg)
                    elif hasattr(pipe, "unet"):
                        quantize_(pipe.unet, fp8_cfg)
                    if verbose:
                        logger.info("Applied FP8 quantization to backbone")
                else:
                    logger.warning("FP8 skipped: no supported Float8WeightOnly config found in torchao")
            except Exception as e:
                logger.warning("FP8 failed: %s — continuing without quantization", e)
    elif strategy.quantization == QuantMode.FP8 and strategy.offload != OffloadMode.NONE:
        if verbose:
            logger.info("Skipping FP8 (incompatible with CPU offload)")

    # --- Offloading
    if strategy.offload == OffloadMode.LAYER_HYBRID:
        # Layer-split hybrid: fill GPU VRAM, overflow to RAM
        pipe.enable_model_cpu_offload()
        if verbose:
            logger.info("Layer hybrid: GPU VRAM + CPU RAM split (model_cpu_offload for diffusers)")
    elif strategy.offload == OffloadMode.SEQUENTIAL_CPU:
        pipe.enable_sequential_cpu_offload()
        if verbose:
            logger.info("Sequential CPU offload: 1 layer at a time (~3GB VRAM)")
    elif strategy.offload == OffloadMode.MODEL_CPU:
        pipe.enable_model_cpu_offload()
        if verbose:
            logger.info("Model CPU offload: components move to GPU on demand")
    elif strategy.offload == OffloadMode.NONE:
        device = _pick_device(hw)
        pipe.to(device)
        if verbose:
            logger.info("Model on %s", device)

    # --- Compile
    do_compile = compile_override if compile_override is not None else strategy.compile
    if do_compile:
        try:
            import triton  # noqa: F401
            backbone = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
            if backbone is not None:
                compiled = torch.compile(backbone, mode=strategy.compile_mode, fullgraph=False)
                if hasattr(pipe, "transformer"):
                    pipe.transformer = compiled
                else:
                    pipe.unet = compiled
                if verbose:
                    logger.info("torch.compile applied (mode=%s)", strategy.compile_mode)
        except ImportError:
            if verbose:
                logger.info("Triton not installed — skipping torch.compile")
        except Exception as e:
            logger.warning("torch.compile failed: %s", e)


def _pick_device(hw: HardwareProfile) -> str:
    if hw.accelerator in (Accelerator.CUDA, Accelerator.ROCm):
        return "cuda"
    if hw.accelerator == Accelerator.MPS:
        return "mps"
    return "cpu"


def _is_diffusers_pipeline(obj: Any) -> bool:
    """Check if obj is a diffusers pipeline (has offload methods)."""
    return (
        hasattr(obj, "enable_sequential_cpu_offload")
        and hasattr(obj, "enable_model_cpu_offload")
    )


def _estimate_model_size(pipe: Any) -> float:
    """Estimate total model size in GB from a diffusers pipeline or model."""
    if _is_diffusers_pipeline(pipe):
        total = 0.0
        for name in ["transformer", "unet", "text_encoder", "text_encoder_2", "vae"]:
            component = getattr(pipe, name, None)
            if component is not None:
                total += _count_params_gb(component)
        return max(total, 1.0)
    return _count_params_gb(pipe)


def _count_params_gb(model: Any) -> float:
    """Count parameter memory in GB."""
    total_bytes = 0
    try:
        for p in model.parameters():
            total_bytes += p.numel() * p.element_size()
    except Exception:
        pass
    return total_bytes / (1024 ** 3)


def _load_fp8_config():
    """Return a Float8WeightOnly config instance, handling torchao API changes across versions."""
    for name in ("Float8WeightOnlyConfig", "Float8WeightOnlyQuantizationConfig"):
        try:
            import importlib
            mod = importlib.import_module("torchao.quantization")
            cls = getattr(mod, name, None)
            if cls is not None:
                return cls()
        except Exception:
            pass
    return None


class MemoryGuard:
    """Context manager for CUDA memory cleanup between inference steps.

    Usage:
        guard = MemoryGuard(threshold=0.7)
        for prompt in prompts:
            with guard:
                result = pipe(prompt)

    With auto-batching:
        guard = MemoryGuard(threshold=0.7)
        for batch in guard.auto_batch(prompts, pipe):
            results = pipe(batch)  # optimal batch size, no OOM
    """

    def __init__(self, threshold: float = 0.7, verbose: bool = False):
        self.threshold = threshold
        self.verbose = verbose
        self._batch_config = None

    def auto_batch(self, items, pipe_or_model=None, **kwargs):
        """Yield optimally-sized batches with VRAM cleanup between them.

        Combines auto-batch calculation with MemoryGuard cleanup.
        """
        from .batch import auto_batch
        for batch in auto_batch(items, pipe_or_model, cleanup_between=True, **kwargs):
            yield batch

    def __enter__(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.synchronize()
        except Exception:
            pass
        return self

    def __exit__(self, *exc):
        try:
            import torch
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                reserved_gb = torch.cuda.memory_reserved() / 1024 ** 3
                total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                usage = reserved_gb / total_gb if total_gb > 0 else 0.0
                if usage > self.threshold:
                    if self.verbose:
                        logger.info("VRAM %.0f%% — deep cleanup", usage * 100)
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                if self.verbose:
                    logger.info("MPS cache cleared")
        except Exception:
            pass
