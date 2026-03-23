"""Tests for strategy caching."""

import json
import subprocess
import sys
import tempfile
import os
from unittest.mock import patch

from overflowml.core.cache import (
    clear_cache,
    hw_fingerprint,
    load_cached_hardware,
    load_cached_model,
    save_cached_hardware,
    save_cached_model,
    show_cache,
)


class TestCache:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self._patch = patch("overflowml.core.cache.CACHE_DIR", __import__("pathlib").Path(self._tmpdir))
        self._patch.start()

    def teardown_method(self):
        self._patch.stop()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_save_and_load_hardware(self):
        save_cached_hardware({"platform": "test", "gpus": []})
        cached = load_cached_hardware()
        assert cached is not None
        assert cached["platform"] == "test"

    def test_cache_miss_on_version_change(self):
        save_cached_hardware({"platform": "test"})
        # Corrupt version
        import pathlib
        p = pathlib.Path(self._tmpdir) / "hardware.json"
        data = json.loads(p.read_text())
        data["_version"] = "old"
        p.write_text(json.dumps(data))
        assert load_cached_hardware() is None

    def test_save_and_load_model(self):
        save_cached_model("test/model", {"param_count": 7000000000})
        cached = load_cached_model("test/model")
        assert cached is not None
        assert cached["param_count"] == 7000000000

    def test_model_cache_miss_different_id(self):
        save_cached_model("test/a", {"param_count": 1})
        assert load_cached_model("test/b") is None

    def test_clear_cache(self):
        save_cached_hardware({"test": True})
        save_cached_model("m", {"test": True})
        count = clear_cache()
        assert count >= 2
        assert load_cached_hardware() is None

    def test_show_cache(self):
        save_cached_hardware({"test": True})
        entries = show_cache()
        assert len(entries) >= 1
        assert entries[0]["fresh"] is True

    def test_hw_fingerprint(self):
        fp1 = hw_fingerprint({"platform": "Linux", "gpus": [{"name": "A100", "total_vram_gb": 80}]})
        fp2 = hw_fingerprint({"platform": "Linux", "gpus": [{"name": "A100", "total_vram_gb": 80}]})
        fp3 = hw_fingerprint({"platform": "Linux", "gpus": [{"name": "RTX 4090", "total_vram_gb": 24}]})
        assert fp1 == fp2
        assert fp1 != fp3


class TestCacheCLI:
    def test_cache_show(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "cache", "show"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0

    def test_cache_clear(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "cache", "clear"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "Cleared" in r.stdout
