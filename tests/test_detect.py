"""Tests for hardware detection — including ROCm vs CUDA disambiguation."""

from unittest.mock import MagicMock, patch

from overflowml.detect import Accelerator, _detect_cuda


def _make_torch_mock(hip_version=None, cuda_available=True, num_gpus=1):
    """Build a mock torch module for detection tests."""
    torch = MagicMock()
    torch.cuda.is_available.return_value = cuda_available
    torch.cuda.device_count.return_value = num_gpus
    props = MagicMock()
    props.name = "Test GPU"
    props.total_memory = 16 * (1024 ** 3)
    props.major = 9
    props.minor = 0
    torch.cuda.get_device_properties.return_value = props
    torch.version.hip = hip_version
    return torch


class TestROCmDetection:
    def test_cuda_gpu_detected_as_cuda(self):
        torch_mock = _make_torch_mock(hip_version=None)
        with patch.dict("sys.modules", {"torch": torch_mock}):
            with patch("psutil.virtual_memory") as mem:
                mem.return_value.total = 64 * (1024 ** 3)
                with patch("psutil.cpu_count", return_value=8):
                    hw = _detect_cuda()
        assert hw is not None
        assert hw.accelerator == Accelerator.CUDA

    def test_rocm_gpu_detected_as_rocm(self):
        torch_mock = _make_torch_mock(hip_version="5.7.0")
        with patch.dict("sys.modules", {"torch": torch_mock}):
            with patch("psutil.virtual_memory") as mem:
                mem.return_value.total = 64 * (1024 ** 3)
                with patch("psutil.cpu_count", return_value=8):
                    hw = _detect_cuda()
        assert hw is not None
        assert hw.accelerator == Accelerator.ROCm

    def test_rocm_never_supports_fp8(self):
        torch_mock = _make_torch_mock(hip_version="6.0.0")
        with patch.dict("sys.modules", {"torch": torch_mock}):
            with patch("psutil.virtual_memory") as mem:
                mem.return_value.total = 64 * (1024 ** 3)
                with patch("psutil.cpu_count", return_value=8):
                    hw = _detect_cuda()
        assert hw is not None
        assert hw.supports_fp8 is False

    def test_no_gpu_returns_none(self):
        torch_mock = _make_torch_mock(cuda_available=False)
        with patch.dict("sys.modules", {"torch": torch_mock}):
            hw = _detect_cuda()
        assert hw is None
