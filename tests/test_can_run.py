"""Tests for can_run() CI/CD gating API."""

import json
import subprocess
import sys

from overflowml.core.can_run import can_run
from overflowml.core.types import CanRunResult


class TestCanRun:
    def test_small_model_fits(self):
        result = can_run(5.0)
        assert isinstance(result, CanRunResult)
        assert result.ok is True

    def test_huge_model_needs_disk(self):
        # 10000GB exceeds 190GB RAM even with INT4 (3000GB) so disk offload
        result = can_run(100000.0)
        assert result.ok is False

    def test_max_offload_none_rejects_offload(self):
        # 200GB model won't fit without offload on any consumer GPU
        result = can_run(200.0, max_offload="none")
        assert result.ok is False

    def test_returns_hardware_info(self):
        result = can_run(10.0)
        assert result.detected_ram_gb > 0

    def test_recommended_strategy_set(self):
        result = can_run(10.0)
        if result.ok:
            assert result.recommended_strategy is not None


class TestCanRunCLI:
    def test_can_run_basic(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "can-run", "10"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "YES" in r.stdout

    def test_can_run_json(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "can-run", "10", "--json"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "ok" in data
        assert "reason" in data

    def test_can_run_huge_model_exits_1(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "can-run", "100000"],
            capture_output=True, text=True,
        )
        assert r.returncode == 1
        assert "NO" in r.stdout

    def test_can_run_max_offload(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "can-run", "10", "--max-offload", "none"],
            capture_output=True, text=True,
        )
        # May pass or fail depending on hardware
        assert r.returncode in (0, 1)

    def test_can_run_in_help(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "--help"],
            capture_output=True, text=True,
        )
        assert "can-run" in r.stdout


class TestCanRunValidation:
    def test_invalid_max_offload_raises(self):
        import pytest
        with pytest.raises(ValueError):
            can_run(5.0, max_offload="modelcpu")

    def test_expert_offload_accepted(self):
        from overflowml.core.types import GPUInfo, HardwareInfo
        hw = HardwareInfo(gpus=[GPUInfo(name="t", total_vram_gb=24, backend="cuda")], total_ram_gb=64)
        assert can_run(5.0, hw, max_offload="expert_offload").ok is True

    def test_non_positive_or_nan_size_raises(self):
        import pytest
        for bad in (0.0, -5.0, float("nan")):
            with pytest.raises(ValueError):
                can_run(bad)

    def test_prequantized_model_is_not_requantized(self):
        from unittest.mock import patch
        from overflowml.core.types import GPUInfo, HardwareInfo, ModelInfo
        # 20GB of AWQ weights on a 16GB card: FP8/INT4 "fits" would be a false YES
        info = ModelInfo(model_id="org/awq", estimated_sizes_gb={"native": 20.0}, prequantized=True)
        hw = HardwareInfo(gpus=[GPUInfo(name="t", total_vram_gb=16, backend="cuda")], total_ram_gb=64,
                          supports_fp8=True)
        with patch("overflowml.inspect.inspect_model", return_value=info):
            result = can_run("org/awq", hw, max_offload="none")
        assert result.ok is False

    def test_unknown_model_is_an_error_not_a_pass(self):
        from unittest.mock import patch
        from overflowml.inspect import ModelSizeUnknown
        with patch("overflowml.inspect.model_estimator.probe_safetensors_params",
                   side_effect=ModelSizeUnknown("typo-org/nope", "not_found")):
            result = can_run("typo-org/nope")
        assert result.ok is False
        assert result.error == "not_found"


class TestCanRunExitCodes:
    """0 = can run, 1 = can't run, 2 = couldn't check."""

    @staticmethod
    def _offline_env(tmp_path):
        import os
        return {**os.environ, "HF_HUB_OFFLINE": "1", "OVERFLOWML_CACHE_DIR": str(tmp_path)}

    def test_json_not_ok_exits_1(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "can-run", "100000", "--json"],
            capture_output=True, text=True,
        )
        assert r.returncode == 1
        assert json.loads(r.stdout)["ok"] is False

    def test_unknown_model_exits_2(self, tmp_path):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "can-run", "typo-org/does-not-exist-xyz"],
            capture_output=True, text=True, env=self._offline_env(tmp_path),
        )
        assert r.returncode == 2
        assert "ERROR" in r.stdout

    def test_unknown_model_json_exits_2(self, tmp_path):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "can-run", "typo-org/does-not-exist-xyz", "--json"],
            capture_output=True, text=True, env=self._offline_env(tmp_path),
        )
        assert r.returncode == 2
        assert json.loads(r.stdout)["error"] == "offline"

    def test_non_positive_size_rejected(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "can-run", "0"],
            capture_output=True, text=True,
        )
        assert r.returncode == 2

    def test_plan_unknown_model_exits_2(self, tmp_path):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "plan", "typo-org/does-not-exist-xyz"],
            capture_output=True, text=True, env=self._offline_env(tmp_path),
        )
        assert r.returncode == 2
        assert "Error:" in r.stderr

    def test_nan_size_rejected(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "can-run", "nan"],
            capture_output=True, text=True,
        )
        assert r.returncode == 2

    def test_plan_json_unknown_model_emits_json_error(self, tmp_path):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "plan", "typo-org/does-not-exist-xyz", "--json"],
            capture_output=True, text=True, env=self._offline_env(tmp_path),
        )
        assert r.returncode == 2
        assert json.loads(r.stdout)["error"] == "offline"

    def test_inspect_unknown_model_exits_2(self, tmp_path):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "inspect", "typo-org/does-not-exist-xyz"],
            capture_output=True, text=True, env=self._offline_env(tmp_path),
        )
        assert r.returncode == 2

    def test_load_unknown_model_exits_2(self, tmp_path):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "load", "typo-org/does-not-exist-xyz"],
            capture_output=True, text=True, env=self._offline_env(tmp_path),
        )
        assert r.returncode == 2
        assert "Error:" in r.stderr

    def test_plan_assume_size_zero_rejected(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "plan", "10", "--assume-size-gb", "0"],
            capture_output=True, text=True,
        )
        assert r.returncode == 2
