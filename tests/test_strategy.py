"""Tests for the strategy engine."""

import pytest
from overflowml.detect import Accelerator, HardwareProfile
from overflowml.strategy import OffloadMode, QuantMode, pick_strategy


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


class TestModelFitsInVRAM:
    def test_small_model_no_offload(self):
        hw = make_hw(gpu_vram_gb=24)
        s = pick_strategy(hw, model_size_gb=10)
        assert s.offload == OffloadMode.NONE
        assert s.quantization == QuantMode.NONE

    def test_model_barely_fits(self):
        hw = make_hw(gpu_vram_gb=24)
        s = pick_strategy(hw, model_size_gb=20)  # 20 * 1.15 = 23 < 24
        assert s.offload == OffloadMode.NONE

    def test_model_too_large_for_vram(self):
        hw = make_hw(gpu_vram_gb=24)
        s = pick_strategy(hw, model_size_gb=22)  # 22 * 1.15 = 25.3 > 24
        assert s.offload != OffloadMode.NONE or s.quantization != QuantMode.NONE


class TestFP8Strategy:
    def test_fp8_makes_model_fit(self):
        hw = make_hw(gpu_vram_gb=24, supports_fp8=True)
        s = pick_strategy(hw, model_size_gb=35)  # 35 * 0.55 * 1.15 = 22.1 < 24
        assert s.quantization == QuantMode.FP8
        assert s.offload == OffloadMode.NONE

    def test_no_fp8_without_support(self):
        hw = make_hw(gpu_vram_gb=24, supports_fp8=False)
        s = pick_strategy(hw, model_size_gb=35)
        assert s.quantization != QuantMode.FP8

    def test_no_fp8_when_disallowed(self):
        hw = make_hw(gpu_vram_gb=24, supports_fp8=True)
        s = pick_strategy(hw, model_size_gb=35, allow_quantization=False)
        assert s.quantization == QuantMode.NONE


class TestCPUOffload:
    def test_sequential_offload_large_model(self):
        hw = make_hw(gpu_vram_gb=24, system_ram_gb=128)
        s = pick_strategy(hw, model_size_gb=40)  # way too big for VRAM
        assert s.offload == OffloadMode.SEQUENTIAL_CPU
        assert s.estimated_vram_gb <= 5.0
        assert s.gc_between_steps is True

    def test_rtx5090_with_40gb_model_fp8_available(self):
        """RTX 5090 with FP8 — FP8 makes model fit without offload."""
        hw = make_hw(gpu_vram_gb=32, system_ram_gb=194, os="Windows", supports_fp8=True)
        s = pick_strategy(hw, model_size_gb=40)
        assert s.quantization == QuantMode.FP8
        assert s.offload == OffloadMode.NONE
        assert s.estimated_vram_gb < 32

    def test_rtx5090_with_40gb_model_no_fp8(self):
        """RTX 5090 without FP8 — model_cpu_offload (components fit individually)."""
        hw = make_hw(gpu_vram_gb=32, system_ram_gb=194, os="Windows", supports_fp8=False)
        s = pick_strategy(hw, model_size_gb=40)
        # 40 <= 32*1.5=48, so individual components fit → model_cpu_offload
        assert s.offload == OffloadMode.MODEL_CPU
        assert s.gc_between_steps is True

    def test_massive_model_needs_sequential(self):
        """Model too large even for component offload."""
        hw = make_hw(gpu_vram_gb=24, system_ram_gb=128, supports_fp8=False)
        s = pick_strategy(hw, model_size_gb=80)  # 80 > 24*1.5=36
        assert s.offload == OffloadMode.SEQUENTIAL_CPU
        assert s.estimated_vram_gb <= 5.0
        assert any("attention_slicing" in w for w in s.warnings)


class TestAppleSilicon:
    def test_model_fits_unified(self):
        hw = make_hw(
            accelerator=Accelerator.MPS,
            gpu_vram_gb=128, system_ram_gb=128,
            unified_memory=True, supports_fp8=False,
        )
        s = pick_strategy(hw, model_size_gb=40)
        assert s.offload == OffloadMode.NONE  # unified memory, no offload needed
        assert s.estimated_vram_gb == 40.0

    def test_model_exceeds_unified(self):
        hw = make_hw(
            accelerator=Accelerator.MPS,
            gpu_vram_gb=16, system_ram_gb=16,
            unified_memory=True, supports_fp8=False,
        )
        s = pick_strategy(hw, model_size_gb=40)  # 40 > 12 (75% of 16)
        assert s.offload == OffloadMode.SEQUENTIAL_CPU
        assert len(s.warnings) > 0

    def test_mlx_detection(self):
        hw = make_hw(
            accelerator=Accelerator.MLX,
            gpu_vram_gb=192, system_ram_gb=192,
            unified_memory=True, supports_fp8=False,
        )
        s = pick_strategy(hw, model_size_gb=80)
        assert s.offload == OffloadMode.NONE
        assert any("unified" in n.lower() for n in s.notes)


class TestROCm:
    def test_rocm_small_model_fits(self):
        hw = make_hw(accelerator=Accelerator.ROCm, gpu_vram_gb=16, supports_fp8=False, supports_bf16=True)
        s = pick_strategy(hw, model_size_gb=10)
        assert s.offload == OffloadMode.NONE
        assert s.quantization == QuantMode.NONE
        assert s.dtype == "bfloat16"

    def test_rocm_no_fp8(self):
        # ROCm never gets FP8 — torchao FP8 is CUDA-only
        hw = make_hw(accelerator=Accelerator.ROCm, gpu_vram_gb=16, supports_fp8=False)
        s = pick_strategy(hw, model_size_gb=20)
        assert s.quantization != QuantMode.FP8

    def test_rocm_large_model_falls_back_to_offload(self):
        hw = make_hw(accelerator=Accelerator.ROCm, gpu_vram_gb=16, system_ram_gb=64, supports_fp8=False)
        s = pick_strategy(hw, model_size_gb=40)
        assert s.offload in (OffloadMode.MODEL_CPU, OffloadMode.SEQUENTIAL_CPU)

    def test_rocm_dtype_no_bf16(self):
        hw = make_hw(accelerator=Accelerator.ROCm, gpu_vram_gb=16, supports_fp8=False, supports_bf16=False)
        s = pick_strategy(hw, model_size_gb=5)
        assert s.dtype == "float16"


class TestEdgeCases:
    def test_zero_vram(self):
        hw = make_hw(accelerator=Accelerator.CPU, gpu_vram_gb=0, system_ram_gb=32)
        s = pick_strategy(hw, model_size_gb=10)
        # Should not crash

    def test_huge_model_tiny_everything(self):
        hw = make_hw(gpu_vram_gb=4, system_ram_gb=8, supports_fp8=False)
        s = pick_strategy(hw, model_size_gb=100)
        # Should pick disk offload or extreme quantization
        assert s.offload in (OffloadMode.SEQUENTIAL_CPU, OffloadMode.DISK)

    def test_forced_strategy(self):
        hw = make_hw(gpu_vram_gb=80)
        s = pick_strategy(
            hw, model_size_gb=10,
            force_offload=OffloadMode.SEQUENTIAL_CPU,
            force_quant=QuantMode.FP8,
        )
        assert s.offload == OffloadMode.SEQUENTIAL_CPU
        assert s.quantization == QuantMode.FP8
