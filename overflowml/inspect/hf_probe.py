"""Hugging Face Hub metadata probing — no weight downloads."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("overflowml")


def probe_safetensors_size(model_id: str) -> Optional[int]:
    """Get total weight size in bytes from safetensors index metadata.

    Uses huggingface_hub API — lightweight, no weight download.
    Returns None if unavailable.
    """
    try:
        from huggingface_hub import model_info
        info = model_info(model_id)
        if info.safetensors and hasattr(info.safetensors, "total"):
            return info.safetensors.total
        # Fallback: check sibling files for safetensors index
        if info.siblings:
            total = 0
            for s in info.siblings:
                if s.rfilename and s.rfilename.endswith(".safetensors"):
                    if s.size:
                        total += s.size
            if total > 0:
                return total
    except Exception as e:
        logger.debug("safetensors probe failed for %s: %s", model_id, e)
    return None


def probe_config(model_id: str, trust_remote_code: bool = False) -> Optional[dict]:
    """Download config.json from HF Hub and return as dict.

    Only downloads the config file, never weight files.
    """
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        return config.to_dict()
    except ImportError:
        logger.debug("transformers not installed — trying huggingface_hub directly")
    except Exception as e:
        logger.debug("AutoConfig failed for %s: %s", model_id, e)

    # Fallback: download config.json directly
    try:
        from huggingface_hub import hf_hub_download
        import json
        path = hf_hub_download(model_id, "config.json")
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.debug("config.json download failed for %s: %s", model_id, e)
    return None
