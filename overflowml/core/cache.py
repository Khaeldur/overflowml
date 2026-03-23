"""Strategy caching — skip repeated hardware detection and planning."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("overflowml")

CACHE_DIR = Path(os.environ.get("OVERFLOWML_CACHE_DIR", os.path.expanduser("~/.cache/overflowml")))
CACHE_TTL_SECONDS = 3600  # 1 hour default


def _cache_dir() -> Path:
    d = CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_key(*parts: str) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _version_tag() -> str:
    from .. import __version__
    return __version__


# --- Hardware cache ---

def load_cached_hardware() -> Optional[dict]:
    """Load cached hardware profile if fresh."""
    path = _cache_dir() / "hardware.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if data.get("_version") != _version_tag():
            return None
        age = time.time() - data.get("_timestamp", 0)
        if age > CACHE_TTL_SECONDS:
            return None
        return data
    except Exception:
        return None


def save_cached_hardware(hw_dict: dict) -> None:
    """Save hardware profile to cache."""
    hw_dict["_version"] = _version_tag()
    hw_dict["_timestamp"] = time.time()
    path = _cache_dir() / "hardware.json"
    try:
        path.write_text(json.dumps(hw_dict, indent=2, default=str))
    except Exception as e:
        logger.debug("Failed to write hardware cache: %s", e)


# --- Model cache ---

def load_cached_model(model_id: str) -> Optional[dict]:
    """Load cached model info if fresh."""
    key = _make_key("model", model_id)
    path = _cache_dir() / f"model_{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if data.get("_version") != _version_tag():
            return None
        age = time.time() - data.get("_timestamp", 0)
        if age > CACHE_TTL_SECONDS:
            return None
        return data
    except Exception:
        return None


def save_cached_model(model_id: str, model_dict: dict) -> None:
    """Save model info to cache."""
    model_dict["_version"] = _version_tag()
    model_dict["_timestamp"] = time.time()
    key = _make_key("model", model_id)
    path = _cache_dir() / f"model_{key}.json"
    try:
        path.write_text(json.dumps(model_dict, indent=2, default=str))
    except Exception as e:
        logger.debug("Failed to write model cache: %s", e)


# --- Plan cache ---

def load_cached_plan(model_id: str, hw_fingerprint: str) -> Optional[dict]:
    """Load cached plan result if fresh."""
    key = _make_key("plan", model_id, hw_fingerprint)
    path = _cache_dir() / f"plan_{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if data.get("_version") != _version_tag():
            return None
        age = time.time() - data.get("_timestamp", 0)
        if age > CACHE_TTL_SECONDS:
            return None
        return data
    except Exception:
        return None


def save_cached_plan(model_id: str, hw_fingerprint: str, plan_dict: dict) -> None:
    """Save plan result to cache."""
    plan_dict["_version"] = _version_tag()
    plan_dict["_timestamp"] = time.time()
    key = _make_key("plan", model_id, hw_fingerprint)
    path = _cache_dir() / f"plan_{key}.json"
    try:
        path.write_text(json.dumps(plan_dict, indent=2, default=str))
    except Exception as e:
        logger.debug("Failed to write plan cache: %s", e)


# --- Cache management ---

def clear_cache() -> int:
    """Clear all cached data. Returns number of files removed."""
    d = _cache_dir()
    count = 0
    for f in d.glob("*.json"):
        try:
            f.unlink()
            count += 1
        except Exception:
            pass
    return count


def show_cache() -> list[dict]:
    """List cached entries with metadata."""
    d = _cache_dir()
    entries = []
    for f in sorted(d.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            age = time.time() - data.get("_timestamp", 0)
            entries.append({
                "file": f.name,
                "version": data.get("_version", "?"),
                "age_seconds": int(age),
                "fresh": age < CACHE_TTL_SECONDS,
            })
        except Exception:
            entries.append({"file": f.name, "error": "unreadable"})
    return entries


def hw_fingerprint(hw_dict: dict) -> str:
    """Generate a fingerprint for hardware state."""
    parts = [
        hw_dict.get("platform", ""),
        str(hw_dict.get("total_ram_gb", 0)),
        str(hw_dict.get("torch_version", "")),
    ]
    gpus = hw_dict.get("gpus", [])
    for g in gpus:
        if isinstance(g, dict):
            parts.append(f"{g.get('name', '')}:{g.get('total_vram_gb', 0)}")
    return _make_key(*parts)
