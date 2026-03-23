"""Tests for CLI — help text, detect, plan, benchmark output."""

import subprocess
import sys


class TestCLIHelp:
    def test_main_help(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "--help"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "Examples:" in r.stdout
        assert "overflowml detect" in r.stdout

    def test_plan_help(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "plan", "--help"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "model_size" in r.stdout

    def test_detect_runs(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "detect"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "Accelerator:" in r.stdout

    def test_plan_basic(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "plan", "10"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "Recommended Strategy" in r.stdout or "Reasoning" in r.stdout

    def test_plan_rejects_zero(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "plan", "0"],
            capture_output=True, text=True,
        )
        assert r.returncode != 0
        assert "positive" in r.stderr

    def test_plan_rejects_negative(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "plan", "--", "-5"],
            capture_output=True, text=True,
        )
        assert r.returncode != 0

    def test_benchmark_runs(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "benchmark"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "Benchmark" in r.stdout
        assert "Legend:" in r.stdout

    def test_benchmark_custom(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "benchmark", "--custom", "7", "140"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "Custom (7GB)" in r.stdout
        assert "Custom (140GB)" in r.stdout

    def test_module_execution(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "--help"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0


class TestCLIValidation:
    def test_benchmark_rejects_zero_custom(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "benchmark", "--custom", "0"],
            capture_output=True, text=True,
        )
        assert r.returncode != 0
        assert "positive" in r.stderr

    def test_benchmark_rejects_negative_custom(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "benchmark", "--custom", "--", "-5"],
            capture_output=True, text=True,
        )
        assert r.returncode != 0

    def test_moe_rejects_active_gt_total(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "plan", "120", "--moe", "100", "200", "128", "8"],
            capture_output=True, text=True,
        )
        assert r.returncode != 0
        assert "active_params_b" in r.stderr

    def test_moe_rejects_active_experts_gt_total(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "plan", "120", "--moe", "120", "12", "8", "16"],
            capture_output=True, text=True,
        )
        assert r.returncode != 0
        assert "active_experts" in r.stderr


class TestNewCLICommands:
    def test_plan_compare(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "plan", "40", "--compare"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "Viable Strategies" in r.stdout

    def test_plan_json(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "plan", "10", "--json"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        import json
        data = json.loads(r.stdout)
        assert "recommended" in data
        assert "strategies" in data
        assert "explanation" in data

    def test_plan_reasoning_shown(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "plan", "40"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "Reasoning" in r.stdout

    def test_plan_compare_shows_rejected(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "plan", "40", "--compare"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        # Should show at least one strategy
        assert "#" in r.stdout or "Rejected" in r.stdout

    def test_inspect_help(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "inspect", "--help"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "model_id" in r.stdout

    def test_help_shows_new_commands(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "--help"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "inspect" in r.stdout
        assert "doctor" in r.stdout


class TestPlanMoE:
    def test_plan_with_moe(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "plan", "120", "--moe", "120", "12", "128", "8"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "MoE:" in r.stdout
        assert "Sparsity:" in r.stdout
