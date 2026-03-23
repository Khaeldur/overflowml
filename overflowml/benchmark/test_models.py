"""Registry of small test models for real inference benchmarks.

These models are chosen to be small enough to download quickly (<2GB)
while still exercising real inference paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class BenchModel:
    model_id: str
    task: Literal["text-generation", "text2text-generation"]
    size_gb: float
    prompt: str
    max_new_tokens: int = 32
    description: str = ""


TEXT_MODELS = [
    BenchModel(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        size_gb=2.2,
        prompt="What is machine learning in one sentence?",
        max_new_tokens=50,
        description="TinyLlama 1.1B — smallest viable causal LM",
    ),
    BenchModel(
        model_id="microsoft/phi-1_5",
        task="text-generation",
        size_gb=2.8,
        prompt="def fibonacci(n):",
        max_new_tokens=64,
        description="Phi-1.5 — small code model",
    ),
]

# Default: smallest model for quick benchmark
DEFAULT_MODEL = TEXT_MODELS[0]
