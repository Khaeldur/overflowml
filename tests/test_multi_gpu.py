"""Tests for multi-GPU detection and strategy."""

import pytest
from overflowml.detect import Accelerator, HardwareProfile
from overflowml.strategy import (
    DistributionMode, OffloadMode, QuantMode, pick_strategy,
)


def make_multi_gpu(num_gpus=2, vram_per_gpu=24.0, ram=64.0, **kwargs):
    defaults = {
        "accelerator": Accelerator.CUDA,
        "gpu_name": "Test GPU",
        "gpu_vram_gb": vram_per_gpu,
        "system_ram_gb": ram,
        "unified_memory": False,
        "os": "Linux",
        "cpu_cores": 16,
        "supports_bf16": True,
        "supports_fp8": True,
        "num_gpus": num_gpus,
        "gpu_names": ["Test GPU"] * num_gpus,
        "gpu_vram_gbs": [vram_per_gpu] * num_gpus,
        "total_gpu_vram_gb": vram_per_gpu * num_gpus,
    }
    defaults.update(kwargs)
    return HardwareProfile(**defaults)


def make_single_gpu(vram=24.0, ram=64.0, **kwargs):
    defaults = {
        "accelerator": Accelerator.CUDA,
        "gpu_name": "Test GPU",
        "gpu_vram_gb": vram,
        "system_ram_gb": ram,
        "unified_memory": False,
        "os": "Linux",
        "cpu_cores": 8,
        "supports_bf16": True,
        "supports_fp8": True,
    }
    defaults.update(kwargs)
    return HardwareProfile(**defaults)


class TestHardwareProfileMultiGPU:
    def test_single_gpu_defaults(self):
        hw = make_single_gpu(vram=24)
        assert hw.num_gpus == 1
        assert hw.total_gpu_vram_gb == 24.0

    def test_multi_gpu_total_vram(self):
        hw = make_multi_gpu(num_gpus=4, vram_per_gpu=80)
        assert hw.total_gpu_vram_gb == 320.0
        assert hw.num_gpus == 4

    def test_effective_memory_multi_gpu(self):
        hw = make_multi_gpu(num_gpus=2, vram_per_gpu=24, ram=64)
        assert hw.effective_memory_gb == 48.0 + (64 - 16)  # 2x24 + overflow

    def test_summary_multi_gpu(self):
        hw = make_multi_gpu(num_gpus=2, vram_per_gpu=32)
        summary = hw.summary()
        assert "2x" in summary
        assert "64GB total" in summary

    def test_summary_single_gpu(self):
        hw = make_single_gpu(vram=32)
        summary = hw.summary()
        assert "GPU:" in summary
        assert "2x" not in summary


class TestMultiGPUDistribution:
    def test_model_fits_single_gpu_no_distribution(self):
        """If model fits on 1 GPU, don't distribute."""
        hw = make_multi_gpu(num_gpus=2, vram_per_gpu=24)
        s = pick_strategy(hw, model_size_gb=16)
        assert s.distribution == DistributionMode.NONE
        assert s.num_gpus_used == 1
        assert s.offload == OffloadMode.NONE

    def test_model_fits_across_2_gpus(self):
        """40GB model on 2x24GB = 48GB total → distribute."""
        hw = make_multi_gpu(num_gpus=2, vram_per_gpu=24)
        s = pick_strategy(hw, model_size_gb=40)
        assert s.distribution == DistributionMode.DEVICE_MAP_AUTO
        assert s.num_gpus_used == 2
        assert s.offload == OffloadMode.NONE

    def test_multi_gpu_estimated_vram_per_gpu(self):
        """Estimated VRAM should be per-GPU, not total."""
        hw = make_multi_gpu(num_gpus=2, vram_per_gpu=24)
        s = pick_strategy(hw, model_size_gb=40)
        assert s.estimated_vram_gb < 24  # per GPU

    def test_fp8_multi_gpu(self):
        """Large model needs FP8 + multi-GPU."""
        hw = make_multi_gpu(num_gpus=4, vram_per_gpu=80)
        s = pick_strategy(hw, model_size_gb=500)
        # 500 * 0.55 = 275, * 1.15 = 316 < 320 (4*80)
        assert s.quantization == QuantMode.FP8
        assert s.distribution == DistributionMode.DEVICE_MAP_AUTO
        assert s.num_gpus_used == 4

    def test_8x_h100_nemotron_340b(self):
        """Nemotron-4 340B (680GB) on 8x H100 80GB = 640GB → needs FP8."""
        hw = make_multi_gpu(num_gpus=8, vram_per_gpu=80, ram=512)
        s = pick_strategy(hw, model_size_gb=680)
        # 680GB doesn't fit in 640GB raw, FP8 = 374GB fits
        assert s.distribution == DistributionMode.DEVICE_MAP_AUTO
        assert s.quantization == QuantMode.FP8
        assert s.num_gpus_used == 8

    def test_8x_h100_llama_405b(self):
        """Llama-3.1-405B (810GB) on 8x H100 → FP8 + distribute."""
        hw = make_multi_gpu(num_gpus=8, vram_per_gpu=80, ram=1024)
        s = pick_strategy(hw, model_size_gb=810)
        # 810 * 0.55 * 1.15 = 512 < 640
        assert s.distribution == DistributionMode.DEVICE_MAP_AUTO
        assert s.quantization == QuantMode.FP8

    def test_multi_gpu_plus_cpu_offload(self):
        """Model too big even for multi-GPU FP8 → distribute + CPU offload."""
        hw = make_multi_gpu(num_gpus=2, vram_per_gpu=24, ram=256)
        s = pick_strategy(hw, model_size_gb=140)
        # 140 * 1.15 = 161 > 48, FP8: 77 * 1.15 = 88.5 > 48
        # Falls through to multi-GPU + CPU offload
        assert s.distribution == DistributionMode.DEVICE_MAP_AUTO
        assert s.offload == OffloadMode.MODEL_CPU
        assert s.num_gpus_used == 2

    def test_multi_gpu_fp8_disabled(self):
        """Multi-GPU without quantization falls through to distribution."""
        hw = make_multi_gpu(num_gpus=2, vram_per_gpu=24)
        s = pick_strategy(hw, model_size_gb=40, allow_quantization=False)
        assert s.distribution == DistributionMode.DEVICE_MAP_AUTO
        assert s.quantization == QuantMode.NONE


class TestMultiGPURegressions:
    def test_single_gpu_unchanged(self):
        """Existing single-GPU behavior must not change."""
        hw = make_single_gpu(vram=24)
        s = pick_strategy(hw, model_size_gb=10)
        assert s.distribution == DistributionMode.NONE
        assert s.num_gpus_used == 1
        assert s.offload == OffloadMode.NONE

    def test_single_gpu_fp8_unchanged(self):
        hw = make_single_gpu(vram=24)
        s = pick_strategy(hw, model_size_gb=35)
        assert s.quantization == QuantMode.FP8
        assert s.offload == OffloadMode.NONE
        assert s.distribution == DistributionMode.NONE

    def test_single_gpu_sequential_unchanged(self):
        hw = make_single_gpu(vram=24, ram=128)
        s = pick_strategy(hw, model_size_gb=80)
        assert s.offload == OffloadMode.SEQUENTIAL_CPU
        assert s.distribution == DistributionMode.NONE

    def test_apple_silicon_ignores_multi_gpu(self):
        """Apple Silicon uses unified memory, no distribution."""
        hw = HardwareProfile(
            accelerator=Accelerator.MPS,
            gpu_name="Apple M4 Max",
            gpu_vram_gb=128, system_ram_gb=128,
            unified_memory=True, supports_bf16=True,
            supports_fp8=False,
        )
        s = pick_strategy(hw, model_size_gb=40)
        assert s.distribution == DistributionMode.NONE
        assert s.offload == OffloadMode.NONE


class TestStrategyNotes:
    def test_multi_gpu_notes(self):
        hw = make_multi_gpu(num_gpus=4, vram_per_gpu=80)
        s = pick_strategy(hw, model_size_gb=200)
        assert any("4 GPUs" in n for n in s.notes)

    def test_strategy_summary_shows_distribution(self):
        hw = make_multi_gpu(num_gpus=2, vram_per_gpu=24)
        s = pick_strategy(hw, model_size_gb=40)
        summary = s.summary()
        assert "Distribution" in summary
        assert "2 GPUs" in summary
