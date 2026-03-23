"""Architecture registry — maps model architectures to task families and param formulas."""

from __future__ import annotations

CAUSAL_LM_PATTERNS = [
    "causallm", "gpt", "llama", "mistral", "qwen", "gemma", "phi",
    "falcon", "mpt", "opt", "bloom", "codegen", "starcoder", "deepseek",
    "mixtral", "cohere", "command",
]

SEQ2SEQ_PATTERNS = ["seq2seq", "conditional", "t5", "bart", "mbart", "pegasus"]

DIFFUSERS_PATTERNS = ["unet", "dit", "flux", "stable-diffusion", "sdxl"]

EMBEDDING_PATTERNS = ["embedding", "sentence-transformer", "bge", "e5"]


def classify_task(architecture: str, model_id: str = "") -> str:
    """Classify a model's task family from architecture name or model ID."""
    combined = (architecture + " " + model_id).lower()

    for pat in DIFFUSERS_PATTERNS:
        if pat in combined:
            return "diffusers"
    for pat in EMBEDDING_PATTERNS:
        if pat in combined:
            return "embedding"
    for pat in SEQ2SEQ_PATTERNS:
        if pat in combined:
            return "seq2seq"
    for pat in CAUSAL_LM_PATTERNS:
        if pat in combined:
            return "causal-lm"
    return "unknown"


def estimate_params_from_config(config: dict) -> tuple[int | None, str]:
    """Estimate param count from HF config dict fields.

    Returns (param_count, source) where source describes estimation method.
    """
    # Explicit param count in config
    for key in ("num_parameters", "n_params", "total_params"):
        val = config.get(key)
        if val is not None and val > 0:
            return int(val), "config.json (explicit)"

    # Architecture-based estimation
    hidden = config.get("hidden_size")
    layers = config.get("num_hidden_layers")
    vocab = config.get("vocab_size")

    if hidden and layers and vocab:
        intermediate = config.get("intermediate_size", hidden * 4)
        num_experts = config.get("num_local_experts") or config.get("num_experts", 1)
        params = (
            vocab * hidden                                           # embeddings
            + layers * (4 * hidden * hidden + 3 * hidden * intermediate * num_experts)  # layers
            + vocab * hidden                                         # lm_head
        )
        return int(params), "config.json (architecture estimate)"

    return None, "unknown"
