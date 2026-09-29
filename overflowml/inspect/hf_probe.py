"""Hugging Face Hub metadata probing — no weight downloads."""

from __future__ import annotations

import json
import logging
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("overflowml")

_REASON_MESSAGES = {
    "not_found": "not found on the Hugging Face Hub (check the ID; private repos need `hf auth login`)",
    "gated": "is gated — accept its license on huggingface.co and run `hf auth login`",
    "offline": "could not be looked up on the Hugging Face Hub",
    "no_metadata": "has no safetensors metadata or usable config.json",
    "missing_dependency": "cannot be looked up: huggingface_hub is not installed (pip install huggingface_hub)",
}

_MAX_HEADER_BYTES = 100 * 1024 * 1024  # safetensors spec limit


class ModelSizeUnknown(ValueError):
    """The model's size could not be determined, so no plan can be trusted.

    reason: not_found | gated | offline | no_metadata | missing_dependency
    """

    def __init__(self, model_id: str, reason: str, detail: str = ""):
        self.model_id = model_id
        self.reason = reason
        msg = f"Model '{model_id}' {_REASON_MESSAGES.get(reason, reason)}"
        if detail:
            msg += f" ({detail})"
        super().__init__(msg + ". Pass the size in GB instead.")


@dataclass
class WeightsProbe:
    params_by_dtype: dict[str, int]  # e.g. {"BF16": 7615616512}
    pipeline: bool = False  # diffusers repo: counts cover only the main component


def probe_safetensors_params(model_id: str) -> Optional[WeightsProbe]:
    """Return per-dtype parameter counts for a Hub repo or local directory.

    Hub repos use the metadata API (no download); local directories read
    safetensors headers only. Returns None when the repo exists but publishes
    no safetensors metadata.

    Raises ModelSizeUnknown when the repo can't be looked up.
    """
    if os.path.isdir(model_id):
        return _local_weights(Path(model_id))
    if os.path.exists(model_id):
        raise ModelSizeUnknown(
            model_id, "no_metadata",
            "a local file; pass the model directory (for GGUF use `overflowml llamacpp`)",
        )

    try:
        from huggingface_hub import model_info
        from huggingface_hub.utils import GatedRepoError, HFValidationError, RepositoryNotFoundError
    except ImportError:
        raise ModelSizeUnknown(model_id, "missing_dependency") from None

    try:
        info = model_info(model_id)
    except GatedRepoError as e:
        raise ModelSizeUnknown(model_id, "gated") from e
    except (RepositoryNotFoundError, HFValidationError) as e:
        raise ModelSizeUnknown(model_id, "not_found") from e
    except Exception as e:
        raise ModelSizeUnknown(model_id, "offline", type(e).__name__) from e

    if info.safetensors and info.safetensors.parameters:
        pipeline = any(s.rfilename == "model_index.json" for s in (info.siblings or []))
        return WeightsProbe(dict(info.safetensors.parameters), pipeline)
    return None


def probe_stored_params(model_id: str) -> Optional[dict[str, int]]:
    """Per-dtype element counts as stored in the safetensors headers.

    The Hub's `parameters` counts packed quantized weights (GPTQ/AWQ I32,
    bnb U8) as unpacked parameters, so multiplying them by the storage width
    overstates the size 2-8x. The headers (fetched with range requests) hold
    the real stored shapes. Returns None if they can't be read.
    """
    if os.path.isdir(model_id):
        local = _local_weights(Path(model_id))
        return local.params_by_dtype if local else None
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars
        was_disabled = are_progress_bars_disabled()
        disable_progress_bars()
        try:
            return dict(HfApi().get_safetensors_metadata(model_id).parameter_count) or None
        finally:
            if not was_disabled:
                enable_progress_bars()
    except Exception as e:
        logger.debug("safetensors header probe failed for %s: %s", model_id, e)
    return None


def _weight_files(path: Path) -> list[Path]:
    """Checkpoint files to count: the index's shards, else all *.safetensors minus duplicates."""
    index = path / "model.safetensors.index.json"
    if index.exists():
        try:
            names = set(json.loads(index.read_text())["weight_map"].values())
            return [path / n for n in sorted(names)]
        except (OSError, ValueError, KeyError) as e:
            logger.debug("unreadable %s: %s", index, e)
    files = sorted(path.glob("*.safetensors"))
    # Mistral-style repos ship consolidated.safetensors next to the HF shards
    shards = [f for f in files if not f.name.startswith("consolidated")]
    return shards or files


def _local_weights(path: Path) -> Optional[WeightsProbe]:
    counts: dict[str, int] = {}
    for f in _weight_files(path):
        try:
            size = f.stat().st_size
            with open(f, "rb") as fh:
                (header_len,) = struct.unpack("<Q", fh.read(8))
                if not 0 < header_len <= min(size - 8, _MAX_HEADER_BYTES):
                    raise ValueError("bad header length")
                header = json.loads(fh.read(header_len))
            for name, tensor in header.items():
                if name == "__metadata__":
                    continue
                counts[tensor["dtype"]] = counts.get(tensor["dtype"], 0) + math.prod(tensor["shape"])
        except (OSError, struct.error, ValueError, KeyError, TypeError) as e:
            raise ModelSizeUnknown(
                str(path), "no_metadata", f"unreadable safetensors header in {f.name} (git-lfs pointer?)",
            ) from e
    if not counts:
        return None
    return WeightsProbe(counts, pipeline=(path / "model_index.json").exists())


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

    # Fallback: read config.json directly
    try:
        if os.path.isdir(model_id):
            path = os.path.join(model_id, "config.json")
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(model_id, "config.json")
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.debug("config.json download failed for %s: %s", model_id, e)
    return None
