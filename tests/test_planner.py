"""Tests for the planning engine."""

from unittest.mock import patch

from overflowml.core.types import GPUInfo, HardwareInfo, PlanResult, StrategyCandidate
from overflowml.core.planner import plan, _generate_candidates
from overflowml.core.explain import build_explanation


def make_hw(**kwargs):
    defaults = dict(
        gpus=[GPUInfo(name="Test GPU", total_vram_gb=24, backend="cuda", device_index=0)],
        total_ram_gb=64, torch_version="2.5.0", torch_cuda_available=True,
        platform="Linux-x86_64", cpu_cores=8, supports_bf16=True, supports_fp8=True,
    )
    defaults.update(kwargs)
    return HardwareInfo(**defaults)


class TestPlan:
    def test_plan_numeric_size(self):
        result = plan(10.0)
        assert isinstance(result, PlanResult)
        assert result.recommended is not None
        assert result.explanation

    def test_plan_compare_returns_multiple(self):
        result = plan(40.0, compare=True)
        assert len(result.strategies) >= 2

    def test_plan_single_returns_viable_only(self):
        result = plan(10.0, compare=False)
        assert all(s.viable for s in result.strategies)

    def test_plan_recommended_is_tagged(self):
        result = plan(40.0, compare=True)
        recommended = [s for s in result.strategies if s.recommended]
        assert len(recommended) == 1

    @patch("overflowml.inspect.model_estimator.inspect_model")
    def test_plan_model_id(self, mock_inspect):
        from overflowml.core.types import ModelInfo
        mock_inspect.return_value = ModelInfo(
            model_id="test/model",
            estimated_sizes_gb={"fp16": 16.0},
            confidence="high",
            source="test",
        )
        result = plan("test/model")
        assert result.model is not None
        assert result.recommended is not None


class TestGenerateCandidates:
    def test_deduplication(self):
        from overflowml.detect import Accelerator, HardwareProfile
        hw = HardwareProfile(
            accelerator=Accelerator.CUDA, gpu_name="Test", gpu_vram_gb=24,
            system_ram_gb=64, supports_bf16=True, supports_fp8=True,
        )
        candidates = _generate_candidates(hw, 10.0)
        keys = [(c.name, c.estimated_speed) for c in candidates]
        assert len(keys) == len(set(c.name for c in candidates))

    def test_sorted_by_speed(self):
        from overflowml.detect import Accelerator, HardwareProfile
        hw = HardwareProfile(
            accelerator=Accelerator.CUDA, gpu_name="Test", gpu_vram_gb=24,
            system_ram_gb=64, supports_bf16=True, supports_fp8=False,
        )
        candidates = _generate_candidates(hw, 40.0)
        viable = [c for c in candidates if c.viable]
        speed_order = {"fastest": 0, "fast": 1, "medium": 2, "slow": 3, "slowest": 4, "unknown": 5}
        speeds = [speed_order.get(c.estimated_speed, 5) for c in viable]
        assert speeds == sorted(speeds)


class TestBuildExplanation:
    def test_includes_hardware(self):
        hw = make_hw()
        rec = StrategyCandidate(name="fp16 direct", notes=["Fits in VRAM"])
        lines = build_explanation(10.0, hw, rec, [rec])
        text = "\n".join(lines)
        assert "Test GPU" in text
        assert "10.0 GB" in text

    def test_includes_rejected(self):
        hw = make_hw()
        rec = StrategyCandidate(name="fp16 offload")
        rej = StrategyCandidate(name="fp16 direct", viable=False, rejection_reason="too big")
        lines = build_explanation(40.0, hw, rec, [rec, rej])
        text = "\n".join(lines)
        assert "Rejected" in text
        assert "too big" in text

    def test_no_gpu(self):
        hw = HardwareInfo(total_ram_gb=32)
        rec = StrategyCandidate(name="cpu only")
        lines = build_explanation(10.0, hw, rec, [rec])
        text = "\n".join(lines)
        assert "No GPU" in text
