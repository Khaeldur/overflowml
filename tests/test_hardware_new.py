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
