"""Tests for optimize.py — device placement, MemoryGuard, parameter estimation."""

from unittest.mock import MagicMock, patch

from overflowml.detect import Accelerator, HardwareProfile
from overflowml.optimize import (
    MemoryGuard,
    _count_params_gb,
    _estimate_model_size,
    _is_diffusers_pipeline,
    _pick_device,
)


def make_hw(**kwargs) -> HardwareProfile:
    defaults = {
        "accelerator": Accelerator.CUDA,
        "gpu_name": "Test GPU",
        "gpu_vram_gb": 24.0,
        "system_ram_gb": 64.0,
        "unified_memory": False,
        "os": "Linux",
        "cpu_cores": 8,
        "supports_bf16": True,
        "supports_fp8": True,
    }
    defaults.update(kwargs)
    return HardwareProfile(**defaults)


class TestPickDevice:
    def test_cuda(self):
        assert _pick_device(make_hw(accelerator=Accelerator.CUDA)) == "cuda"

    def test_rocm(self):
        assert _pick_device(make_hw(accelerator=Accelerator.ROCm)) == "cuda"

    def test_mps(self):
        assert _pick_device(make_hw(accelerator=Accelerator.MPS)) == "mps"

    def test_mlx(self):
        assert _pick_device(make_hw(accelerator=Accelerator.MLX)) == "cpu"

    def test_cpu(self):
        assert _pick_device(make_hw(accelerator=Accelerator.CPU)) == "cpu"


class TestIsDiffusersPipeline:
    def test_real_pipeline_like_object(self):
        obj = MagicMock()
        obj.enable_sequential_cpu_offload = MagicMock()
        obj.enable_model_cpu_offload = MagicMock()
        assert _is_diffusers_pipeline(obj) is True

    def test_plain_model(self):
        obj = MagicMock(spec=[])
        assert _is_diffusers_pipeline(obj) is False


class TestCountParamsGb:
    def test_known_params(self):
        model = MagicMock()
        p1 = MagicMock()
        p1.numel.return_value = 1_000_000
        p1.element_size.return_value = 2  # BF16
        model.parameters.return_value = [p1]
        gb = _count_params_gb(model)
        expected = (1_000_000 * 2) / (1024 ** 3)
        assert abs(gb - expected) < 1e-6

    def test_no_parameters(self):
        model = MagicMock()
        model.parameters.side_effect = AttributeError
        assert _count_params_gb(model) == 0.0


class TestEstimateModelSize:
    def test_pipeline_with_transformer_and_vae(self):
        pipe = MagicMock()
        pipe.enable_sequential_cpu_offload = MagicMock()
        pipe.enable_model_cpu_offload = MagicMock()

        # transformer with 1B params at 2 bytes each = ~1.86 GB
        t_param = MagicMock()
        t_param.numel.return_value = 1_000_000_000
        t_param.element_size.return_value = 2
        pipe.transformer = MagicMock()
        pipe.transformer.parameters.return_value = [t_param]

        # vae with 100M params
        v_param = MagicMock()
        v_param.numel.return_value = 100_000_000
        v_param.element_size.return_value = 2
        pipe.vae = MagicMock()
        pipe.vae.parameters.return_value = [v_param]

        pipe.unet = None
        pipe.text_encoder = None
        pipe.text_encoder_2 = None

        size = _estimate_model_size(pipe)
        assert size > 1.5  # at least the transformer

    def test_standalone_model(self):
        model = MagicMock(spec=["parameters"])
        p = MagicMock()
        p.numel.return_value = 500_000_000
        p.element_size.return_value = 2
        model.parameters.return_value = [p]
        size = _estimate_model_size(model)
        assert size > 0.9


class TestMemoryGuard:
    def test_context_manager_basic(self):
        guard = MemoryGuard(threshold=0.7)
        with guard:
            pass  # should not crash

    def test_threshold_stored(self):
        guard = MemoryGuard(threshold=0.5, verbose=True)
        assert guard.threshold == 0.5
        assert guard.verbose is True

    def test_enter_returns_self(self):
        guard = MemoryGuard()
        result = guard.__enter__()
        assert result is guard
        guard.__exit__(None, None, None)

    def test_exit_calls_gc(self):
        guard = MemoryGuard()
        with patch("overflowml.optimize.gc.collect") as mock_gc:
            guard.__enter__()
            guard.__exit__(None, None, None)
            assert mock_gc.called

    def test_exit_handles_zero_vram_gpu(self):
        """MemoryGuard should not crash if GPU reports 0 total memory."""
        guard = MemoryGuard()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_reserved.return_value = 0
        props = MagicMock()
        props.total_memory = 0  # edge case: 0 VRAM
        mock_torch.cuda.get_device_properties.return_value = props
        with patch.dict("sys.modules", {"torch": mock_torch}):
            guard.__exit__(None, None, None)  # should not raise ZeroDivisionError
