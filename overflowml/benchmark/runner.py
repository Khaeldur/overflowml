"""Benchmark runner — measures real inference performance."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .test_models import DEFAULT_MODEL, TEXT_MODELS, BenchModel

logger = logging.getLogger("overflowml")


@dataclass
class BenchmarkResult:
    model_id: str = ""
    model_size_gb: float = 0.0
    strategy_used: str = ""
    load_time_s: float = 0.0
    warmup_time_s: float = 0.0
    inference_time_s: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    peak_vram_gb: float = 0.0
    output_text: str = ""
    error: str = ""
    predicted_strategy: str = ""
    prediction_matched: bool = False
    notes: list[str] = field(default_factory=list)


def run_benchmark(
    model_id: Optional[str] = None,
    task: str = "text-generation",
    trust_remote_code: bool = False,
    verbose: bool = True,
) -> BenchmarkResult:
    """Run a real inference benchmark.

    Downloads a small test model, runs inference, measures performance,
    and compares against OverflowML's predicted strategy.

    Args:
        model_id: HuggingFace model ID. If None, uses default tiny model.
        task: "text-generation" (default).
        trust_remote_code: Pass to from_pretrained.
        verbose: Print progress.

    Returns:
        BenchmarkResult with timing, throughput, and prediction accuracy.
    """
    result = BenchmarkResult()

    # Pick test model
    test_model = _resolve_model(model_id, task)
    result.model_id = test_model.model_id
    result.model_size_gb = test_model.size_gb

    if verbose:
        logger.info("Benchmark: %s (%s, ~%.1fGB)", test_model.model_id, test_model.task, test_model.size_gb)

    # Get OverflowML's prediction first
    try:
        from ..core.planner import plan
        plan_result = plan(test_model.size_gb)
        if plan_result.recommended:
            result.predicted_strategy = plan_result.recommended.name
            if verbose:
                logger.info("Predicted strategy: %s", result.predicted_strategy)
    except Exception as e:
        result.notes.append(f"Plan prediction failed: {e}")

    # Load model
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        result.error = f"Missing dependency: {e}. Install: pip install overflowml[transformers]"
        return result

    if verbose:
        logger.info("Loading model...")

    t0 = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            test_model.model_id, trust_remote_code=trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            test_model.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )

        # Try to move to GPU
        device = "cpu"
        if torch.cuda.is_available():
            try:
                model = model.cuda()
                device = "cuda"
            except RuntimeError:
                result.notes.append("Model didn't fit in VRAM, running on CPU")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                model = model.to("mps")
                device = "mps"
            except RuntimeError:
                pass

        result.load_time_s = time.perf_counter() - t0
        result.strategy_used = f"fp16 on {device}"

        if verbose:
            logger.info("Loaded in %.1fs on %s", result.load_time_s, device)
    except Exception as e:
        result.error = f"Load failed: {e}"
        result.load_time_s = time.perf_counter() - t0
        return result

    # Warmup
    if verbose:
        logger.info("Warmup run...")
    t0 = time.perf_counter()
    try:
        inputs = tokenizer(test_model.prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=5, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result.warmup_time_s = time.perf_counter() - t0
    except Exception as e:
        result.error = f"Warmup failed: {e}"
        return result

    # Timed inference
    if verbose:
        logger.info("Timed inference (%d tokens)...", test_model.max_new_tokens)
    t0 = time.perf_counter()
    try:
        inputs = tokenizer(test_model.prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=test_model.max_new_tokens,
                do_sample=False,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        result.inference_time_s = time.perf_counter() - t0
        result.tokens_generated = outputs.shape[1] - input_len
        result.tokens_per_second = result.tokens_generated / max(result.inference_time_s, 0.001)
        result.output_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        # Peak VRAM
        if torch.cuda.is_available():
            result.peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        if verbose:
            logger.info("Generated %d tokens in %.2fs (%.1f tok/s)",
                        result.tokens_generated, result.inference_time_s, result.tokens_per_second)
    except Exception as e:
        result.error = f"Inference failed: {e}"
        return result

    # Compare prediction vs reality
    if result.predicted_strategy:
        if "direct" in result.predicted_strategy.lower() and device != "cpu":
            result.prediction_matched = True
            result.notes.append("Prediction matched: model loaded directly on GPU as predicted")
        elif "offload" in result.predicted_strategy.lower() and device == "cpu":
            result.prediction_matched = True
            result.notes.append("Prediction matched: offload strategy used as predicted")
        else:
            result.notes.append(f"Predicted '{result.predicted_strategy}', actual: fp16 on {device}")

    # Cleanup
    del model, tokenizer
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return result


def _resolve_model(model_id: Optional[str], task: str) -> BenchModel:
    if model_id is None:
        return DEFAULT_MODEL
    for m in TEXT_MODELS:
        if m.model_id == model_id:
            return m
    # Custom model
    return BenchModel(
        model_id=model_id,
        task=task,
        size_gb=0,
        prompt="Hello, ",
        max_new_tokens=32,
        description="Custom model",
    )
