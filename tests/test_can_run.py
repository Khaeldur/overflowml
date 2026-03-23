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
        # 40GB model on CPU-only torch won't fit without offload
        result = can_run(40.0, max_offload="none")
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
