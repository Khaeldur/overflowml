"""Tests for core data types."""

from overflowml.core.types import (
    GPUInfo, HardwareInfo, ModelInfo, StrategyCandidate, PlanResult,
    DoctorIssue, DoctorReport, CanRunResult,
)


class TestHardwareInfo:
    def test_total_vram_empty(self):
        hw = HardwareInfo()
        assert hw.total_vram_gb == 0.0
        assert hw.num_gpus == 0
        assert hw.primary_backend == "cpu"

    def test_total_vram_multi_gpu(self):
        hw = HardwareInfo(gpus=[
            GPUInfo(name="A", total_vram_gb=24, backend="cuda", device_index=0),
            GPUInfo(name="B", total_vram_gb=24, backend="cuda", device_index=1),
        ])
        assert hw.total_vram_gb == 48.0
        assert hw.num_gpus == 2
        assert hw.primary_backend == "cuda"


class TestModelInfo:
    def test_defaults(self):
        info = ModelInfo(model_id="test")
        assert info.confidence == "low"
        assert info.task_family == "unknown"
        assert info.estimated_sizes_gb == {}

    def test_with_sizes(self):
        info = ModelInfo(
            model_id="llama-70b",
            param_count=70_000_000_000,
            estimated_sizes_gb={"fp16": 140.0, "int8": 70.0, "int4": 35.0},
            confidence="high",
        )
        assert info.estimated_sizes_gb["fp16"] == 140.0


class TestDoctorReport:
    def test_ok_default(self):
        r = DoctorReport()
        assert r.ok is True

    def test_with_error(self):
        r = DoctorReport(
            ok=False,
            issues=[DoctorIssue(code="test", severity="error", message="bad")],
        )
        assert r.ok is False


class TestStrategyCandidate:
    def test_defaults(self):
        s = StrategyCandidate(name="test")
        assert s.viable is True
        assert s.recommended is False
