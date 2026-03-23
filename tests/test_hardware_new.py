"""Tests for new hardware detection (core.hardware)."""

from overflowml.core.hardware import detect_hardware_info
from overflowml.core.types import HardwareInfo


class TestDetectHardwareInfo:
    def test_returns_hardware_info(self):
        hw = detect_hardware_info()
        assert isinstance(hw, HardwareInfo)
        assert hw.platform != ""

    def test_has_ram(self):
        hw = detect_hardware_info()
        assert hw.total_ram_gb > 0

    def test_cpu_cores(self):
        hw = detect_hardware_info()
        assert hw.cpu_cores > 0

    def test_torch_version_set(self):
        hw = detect_hardware_info()
        assert hw.torch_version is not None  # torch is installed in test env


class TestHardwareInfoToLegacy:
    def test_empty_gpus(self):
        from overflowml.core.hardware import hardware_info_to_legacy
        hw = HardwareInfo(total_ram_gb=32, platform="Linux-x86_64")
        legacy = hardware_info_to_legacy(hw)
        assert legacy.gpu_name == "None"
        assert legacy.gpu_vram_gb == 0.0

    def test_single_gpu(self):
        from overflowml.core.hardware import hardware_info_to_legacy
        from overflowml.core.types import GPUInfo
        hw = HardwareInfo(
            gpus=[GPUInfo(name="RTX 5090", total_vram_gb=32, backend="cuda")],
            total_ram_gb=64, platform="Windows-AMD64",
        )
        legacy = hardware_info_to_legacy(hw)
        assert legacy.gpu_name == "RTX 5090"
        assert legacy.gpu_vram_gb == 32.0
        assert legacy.num_gpus == 1

    def test_multi_gpu(self):
        from overflowml.core.hardware import hardware_info_to_legacy
        from overflowml.core.types import GPUInfo
        hw = HardwareInfo(
            gpus=[
                GPUInfo(name="A100", total_vram_gb=80, backend="cuda", device_index=0),
                GPUInfo(name="A100", total_vram_gb=80, backend="cuda", device_index=1),
            ],
            total_ram_gb=256, platform="Linux-x86_64",
        )
        legacy = hardware_info_to_legacy(hw)
        assert legacy.num_gpus == 2
        assert legacy.total_gpu_vram_gb == 160.0
