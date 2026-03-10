"""Apply optimization strategy to models and pipelines."""

from __future__ import annotations

import gc
import logging
from typing import Any, Optional

from .detect import Accelerator, HardwareProfile, detect_hardware
from .strategy import OffloadMode, QuantMode, Strategy, pick_strategy

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
    if hw is None:
        hw = detect_hardware()

    if model_size_gb is None:
        model_size_gb = _estimate_model_size(pipe)

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
                from torchao.quantization import quantize_, Float8WeightOnlyConfig
                quantize_(model, Float8WeightOnlyConfig())
                if verbose:
                    logger.info("Applied FP8 weight-only quantization")
            except Exception as e:
                logger.warning("FP8 quantization failed: %s", e)

    # Device placement
    if strategy.offload == OffloadMode.NONE:
        device = "cuda" if hw.accelerator == Accelerator.CUDA else "mps" if hw.accelerator == Accelerator.MPS else "cpu"
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
                from torchao.quantization import quantize_, Float8WeightOnlyConfig
                if hasattr(pipe, "transformer"):
                    quantize_(pipe.transformer, Float8WeightOnlyConfig())
                elif hasattr(pipe, "unet"):
                    quantize_(pipe.unet, Float8WeightOnlyConfig())
                if verbose:
                    logger.info("Applied FP8 quantization to backbone")
            except Exception as e:
                logger.warning("FP8 failed: %s — continuing without quantization", e)
    elif strategy.quantization == QuantMode.FP8 and strategy.offload != OffloadMode.NONE:
        if verbose:
            logger.info("Skipping FP8 (incompatible with CPU offload)")

    # --- Offloading
    if strategy.offload == OffloadMode.SEQUENTIAL_CPU:
        pipe.enable_sequential_cpu_offload()
        if verbose:
            logger.info("Sequential CPU offload: 1 layer at a time (~3GB VRAM)")
    elif strategy.offload == OffloadMode.MODEL_CPU:
        pipe.enable_model_cpu_offload()
        if verbose:
            logger.info("Model CPU offload: components move to GPU on demand")
    elif strategy.offload == OffloadMode.NONE:
        device = "cuda" if hw.accelerator == Accelerator.CUDA else "mps"
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


def _estimate_model_size(pipe: Any) -> float:
    """Estimate total model size in GB from a diffusers pipeline."""
    total = 0.0
    for name in ["transformer", "unet", "text_encoder", "text_encoder_2", "vae"]:
        component = getattr(pipe, name, None)
        if component is not None:
            total += _count_params_gb(component)
    return max(total, 1.0)


def _count_params_gb(model: Any) -> float:
    """Count parameter memory in GB."""
    total_bytes = 0
    try:
        for p in model.parameters():
            total_bytes += p.numel() * p.element_size()
    except Exception:
        pass
    return total_bytes / (1024 ** 3)


class MemoryGuard:
    """Context manager for CUDA memory cleanup between inference steps.

    Usage:
        guard = MemoryGuard(threshold=0.7)
        for prompt in prompts:
            with guard:
                result = pipe(prompt)
    """

    def __init__(self, threshold: float = 0.7, verbose: bool = False):
        self.threshold = threshold
        self.verbose = verbose

    def __enter__(self):
        try:
            import torch
            torch.cuda.synchronize()
        except Exception:
            pass
        return self

    def __exit__(self, *exc):
        try:
            import torch
            if not torch.cuda.is_available():
                return

            gc.collect()
            torch.cuda.empty_cache()

            reserved_gb = torch.cuda.memory_reserved() / 1024 ** 3
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            usage = reserved_gb / total_gb

            if usage > self.threshold:
                if self.verbose:
                    logger.info("VRAM %.0f%% — deep cleanup", usage * 100)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
