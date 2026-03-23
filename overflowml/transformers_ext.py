"""HuggingFace transformers integration — load any LLM with optimal strategy."""

from __future__ import annotations

import logging
from typing import Any, Optional

from .detect import Accelerator, HardwareProfile, detect_hardware
from .strategy import DistributionMode, OffloadMode, QuantMode, Strategy, pick_strategy

logger = logging.getLogger("overflowml")


def load_model(
    model_name: str,
    model_size_gb: Optional[float] = None,
    hw: Optional[HardwareProfile] = None,
    strategy: Optional[Strategy] = None,
    *,
    model_class: Optional[Any] = None,
    tokenizer: bool = True,
    prefer_speed: bool = True,
    allow_quantization: bool = True,
    trust_remote_code: bool = False,
    verbose: bool = True,
    **from_pretrained_kwargs,
):
    """Load a HuggingFace model with optimal memory strategy.

    Usage:
        import overflowml

        # Automatic — detects hardware, picks strategy, loads model
        model, tokenizer = overflowml.load_model("meta-llama/Llama-3-70B")

        # With options
        model, tokenizer = overflowml.load_model(
            "meta-llama/Llama-3-70B",
            model_size_gb=140,  # optional: override auto-detection
            prefer_speed=True,
        )

    Args:
        model_name: HuggingFace model ID or local path
        model_size_gb: Model size in GB (BF16). Auto-estimated from config if not provided.
        hw: Hardware profile. Auto-detected if not provided.
        strategy: Pre-computed strategy. Auto-computed if not provided.
        model_class: Specific model class (e.g., AutoModelForCausalLM). Auto-detected if None.
        tokenizer: Also load and return tokenizer.
        prefer_speed: Prefer faster strategies over lower VRAM usage.
        allow_quantization: Allow quantization.
        trust_remote_code: Pass to from_pretrained.
        verbose: Print what's happening.
        **from_pretrained_kwargs: Extra args passed to from_pretrained.

    Returns:
        (model, tokenizer) if tokenizer=True, else just model
    """
    import torch

    if hw is None:
        hw = detect_hardware()

    if model_size_gb is None:
        model_size_gb = _estimate_from_config(model_name, trust_remote_code)

    if strategy is None:
        strategy = pick_strategy(
            hw, model_size_gb,
            prefer_speed=prefer_speed,
            allow_quantization=allow_quantization,
        )

    if verbose:
        logger.info("OverflowML: Loading %s (%.0fGB)", model_name, model_size_gb)
        logger.info("Hardware:\n%s", hw.summary())
        logger.info("Strategy:\n%s", strategy.summary())

    # Build from_pretrained kwargs based on strategy
    kwargs = dict(from_pretrained_kwargs)
    kwargs["trust_remote_code"] = trust_remote_code
    kwargs["low_cpu_mem_usage"] = True

    # Dtype
    if strategy.dtype == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif strategy.dtype == "float16":
        kwargs["torch_dtype"] = torch.float16

    # Quantization via bitsandbytes (widely supported)
    if strategy.quantization == QuantMode.INT4:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if hw.supports_bf16 else torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            if verbose:
                logger.info("INT4 quantization (bitsandbytes NF4)")
        except ImportError:
            logger.warning("bitsandbytes not installed — skipping INT4 quantization")

    elif strategy.quantization == QuantMode.INT8:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            if verbose:
                logger.info("INT8 quantization (bitsandbytes)")
        except ImportError:
            logger.warning("bitsandbytes not installed — skipping INT8 quantization")

    # Device map based on distribution and offload strategy
    if strategy.distribution == DistributionMode.DEVICE_MAP_AUTO:
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = _max_memory_map(hw)
    elif strategy.offload == OffloadMode.NONE:
        kwargs["device_map"] = {"": 0}  # everything on GPU 0
    elif strategy.offload == OffloadMode.MODEL_CPU:
        kwargs["device_map"] = "auto"
    elif strategy.offload == OffloadMode.SEQUENTIAL_CPU:
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = _max_memory_map(hw)
    elif strategy.offload == OffloadMode.DISK:
        import os
        kwargs["device_map"] = "auto"
        kwargs["offload_folder"] = os.environ.get(
            "OVERFLOWML_OFFLOAD_DIR",
            os.path.join(os.path.expanduser("~"), ".cache", "overflowml", "offload"),
        )
        kwargs["max_memory"] = _max_memory_map(hw, reserve_gpu_gb=2)

    # Resolve model class
    if model_class is None:
        model_class = _detect_model_class(model_name, trust_remote_code)

    if verbose:
        logger.info("Loading with device_map=%s ...", kwargs.get("device_map", "default"))

    model = model_class.from_pretrained(model_name, **kwargs)

    # FP8 post-load quantization (when not using offload)
    if (strategy.quantization == QuantMode.FP8
            and strategy.offload == OffloadMode.NONE
            and hw.supports_fp8
            and "quantization_config" not in kwargs):
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

    if verbose:
        logger.info("Model loaded successfully")
        _log_device_map(model)

    if not tokenizer:
        return model

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    return model, tok


def _estimate_from_config(model_name: str, trust_remote_code: bool = False) -> float:
    """Estimate model size from HuggingFace config without downloading weights."""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)

        num_params = getattr(config, "num_parameters", None)
        if num_params is None:
            hidden = getattr(config, "hidden_size", 4096)
            layers = getattr(config, "num_hidden_layers", 32)
            vocab = getattr(config, "vocab_size", 32000)
            intermediate = getattr(config, "intermediate_size", hidden * 4)
            # Rough estimate: embedding + layers * (attn + ffn) + lm_head
            num_params = (
                vocab * hidden  # embeddings
                + layers * (4 * hidden * hidden + 3 * hidden * intermediate)  # layers
                + vocab * hidden  # lm_head
            )

        # BF16 = 2 bytes per param
        size_gb = (num_params * 2) / (1024 ** 3)
        logger.info("Estimated model size: %.1fGB (%.1fB params)", size_gb, num_params / 1e9)
        return max(size_gb, 0.5)
    except Exception as e:
        logger.warning("Could not estimate model size from config: %s — defaulting to 14GB", e)
        return 14.0


def _detect_model_class(model_name: str, trust_remote_code: bool = False):
    """Auto-detect the right AutoModel class for this model."""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        architectures = getattr(config, "architectures", []) or []

        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
        )

        for arch in architectures:
            arch_lower = arch.lower()
            if "causallm" in arch_lower or "gpt" in arch_lower:
                return AutoModelForCausalLM
            if "seq2seq" in arch_lower or "conditional" in arch_lower:
                return AutoModelForSeq2SeqLM
            if "sequenceclassification" in arch_lower:
                return AutoModelForSequenceClassification

        # Default to causal LM (most common)
        return AutoModelForCausalLM
    except Exception:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM


def _max_memory_map(hw: HardwareProfile, reserve_gpu_gb: float = 4) -> dict:
    """Build max_memory dict for accelerate device_map."""
    mem = {}
    if hw.accelerator in (Accelerator.CUDA, Accelerator.ROCm):
        for i in range(hw.num_gpus):
            per_gpu_vram = hw.gpu_vram_gbs[i] if (hw.gpu_vram_gbs and i < len(hw.gpu_vram_gbs)) else hw.gpu_vram_gb
            usable = max(1, per_gpu_vram - reserve_gpu_gb)
            mem[i] = f"{int(usable)}GiB"
    usable_cpu = max(1, hw.system_ram_gb - 16) if hw.system_ram_gb > 0 else 0
    if usable_cpu > 0:
        mem["cpu"] = f"{int(usable_cpu)}GiB"
    return mem


def _log_device_map(model: Any):
    """Log which devices the model layers ended up on."""
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        devices = set(str(v) for v in device_map.values())
        logger.info("Device distribution: %s", ", ".join(sorted(devices)))


def _load_fp8_config():
    """Return a Float8WeightOnly config instance, handling torchao API changes across versions."""
    # torchao <0.9: Float8WeightOnlyConfig
    # torchao 0.9+: same name but may also have Float8WeightOnlyQuantizationConfig alias
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
