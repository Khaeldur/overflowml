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

    # Architecture-based estimation; multimodal configs nest the language model
    cfg = config.get("text_config") or config
    hidden = cfg.get("hidden_size")
    layers = cfg.get("num_hidden_layers")
    vocab = cfg.get("vocab_size") or config.get("vocab_size")

    if hidden and layers and vocab:
        heads = cfg.get("num_attention_heads")
        head_dim = cfg.get("head_dim") or (hidden // heads if heads else None)
        if heads and head_dim:
            kv_heads = cfg.get("num_key_value_heads") or heads
            attn = hidden * head_dim * 2 * (heads + kv_heads)  # q+o, k+v (GQA)
        else:
            attn = 4 * hidden * hidden

        intermediate = cfg.get("intermediate_size") or hidden * 4
        dense_ffn = 3 * hidden * intermediate
        num_experts = cfg.get("num_local_experts") or cfg.get("num_experts") or cfg.get("n_routed_experts") or 1
        if num_experts > 1:
            expert_size = cfg.get("moe_intermediate_size") or intermediate
            shared_size = cfg.get("shared_expert_intermediate_size") or (cfg.get("n_shared_experts") or 0) * expert_size
            moe_ffn = 3 * hidden * (expert_size * num_experts + shared_size)
            dense_layers = min(cfg.get("first_k_dense_replace") or 0, layers)
            ffn_total = dense_layers * dense_ffn + (layers - dense_layers) * moe_ffn
        else:
            ffn_total = layers * dense_ffn

        tied = cfg.get("tie_word_embeddings", config.get("tie_word_embeddings", False))
        embeddings = vocab * hidden * (1 if tied else 2)
        params = embeddings + layers * attn + ffn_total
        return int(params), "config.json (architecture estimate)"

    return None, "unknown"
