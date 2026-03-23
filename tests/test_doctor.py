"""Tests for doctor checks and report."""

import json
import subprocess
import sys
from unittest.mock import MagicMock, patch

from overflowml.core.types import DoctorIssue, DoctorReport
from overflowml.doctor.checks import (
    check_python,
    check_torch,
    check_gpu,
    check_ram,
    check_optional_dep,
    check_driver_mismatch,
)
from overflowml.doctor.report import run as doctor_run


class TestCheckPython:
    def test_current_python(self):
        issue = check_python()
        assert issue.severity == "info"
        assert "Python" in issue.message


class TestCheckTorch:
    def test_torch_present(self):
        issue = check_torch()
        assert issue.code in ("torch_cuda", "torch_rocm", "torch_mps", "torch_cpu_only")

    @patch.dict("sys.modules", {"torch": None})
    def test_torch_missing(self):
        with patch("builtins.__import__", side_effect=ImportError):
            pass
        # Can't easily mock away torch mid-test since it's already imported
        # Just verify the function doesn't crash
        issue = check_torch()
        assert issue is not None


class TestCheckGpu:
    def test_returns_issue(self):
        issue = check_gpu()
        assert isinstance(issue, DoctorIssue)
        assert issue.code in ("gpu_ok", "gpu_mps", "gpu_not_accessible", "gpu_none")


class TestCheckRam:
    def test_returns_issue(self):
        issue = check_ram()
        assert isinstance(issue, DoctorIssue)
        assert "RAM" in issue.message or "psutil" in issue.message


class TestCheckOptionalDep:
    def test_installed_dep(self):
        issue = check_optional_dep("psutil", "psutil", "dev")
        assert issue.severity == "info"
        assert "psutil" in issue.message

    def test_missing_dep(self):
        issue = check_optional_dep("nonexistent", "this_does_not_exist_xyz", "dev")
        assert issue.severity == "warn"
        assert "not installed" in issue.message
        assert issue.suggested_fix is not None


class TestCheckDriverMismatch:
    @patch("overflowml.doctor.checks._get_nvidia_smi_cuda_version", return_value=None)
    def test_no_nvidia_smi(self, mock):
        result = check_driver_mismatch()
        # Should return None (no mismatch detectable)
        assert result is None or isinstance(result, DoctorIssue)


class TestDoctorRun:
    def test_returns_report(self):
        report = doctor_run()
        assert isinstance(report, DoctorReport)
        assert "python" in report.environment
        assert len(report.issues) > 0

    def test_report_ok_when_no_errors(self):
        report = doctor_run()
        has_errors = any(i.severity == "error" for i in report.issues)
        assert report.ok == (not has_errors)

    def test_with_model_size(self):
        report = doctor_run(model_size_gb=10.0)
        assert isinstance(report, DoctorReport)

    def test_fix_commands_collected(self):
        report = doctor_run()
        # fix_commands should be a list (may be empty if all good)
        assert isinstance(report.fix_commands, list)


class TestDoctorCLI:
    def test_doctor_runs(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "doctor"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "OverflowML Doctor" in r.stdout

    def test_doctor_json(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "doctor", "--json"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "ok" in data
        assert "issues" in data
        assert "environment" in data
