"""Tests for transformers integration."""

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


class TestLLMStrategies:
    def test_8b_model_on_24gb(self):
        """Llama-3-8B (16GB BF16) on RTX 4090 — fits in VRAM."""
        hw = make_hw(gpu_vram_gb=24)
        s = pick_strategy(hw, model_size_gb=16)
        assert s.offload == OffloadMode.NONE

    def test_70b_model_on_24gb(self):
        """Llama-3-70B (140GB BF16) on RTX 4090 — needs offload to RAM."""
        hw = make_hw(gpu_vram_gb=24, system_ram_gb=192, supports_fp8=False)
        s = pick_strategy(hw, model_size_gb=140)
        assert s.offload in (OffloadMode.SEQUENTIAL_CPU, OffloadMode.LAYER_HYBRID)

    def test_70b_on_mac_128gb(self):
        """Llama-3-70B on Mac M4 Max (128GB) — fits in unified memory."""
        hw = make_hw(
            accelerator=Accelerator.MPS,
            gpu_vram_gb=128, system_ram_gb=128,
            unified_memory=True, supports_fp8=False,
        )
        # 140GB > 96GB effective — won't fit, needs warning
        s = pick_strategy(hw, model_size_gb=140)
        assert len(s.warnings) > 0

    def test_70b_int4_on_mac_64gb(self):
        """70B at INT4 (~35GB) on Mac 64GB — should fit."""
        hw = make_hw(
            accelerator=Accelerator.MPS,
            gpu_vram_gb=64, system_ram_gb=64,
            unified_memory=True, supports_fp8=False,
        )
        # INT4 halves again: 35GB < 48GB effective
        s = pick_strategy(hw, model_size_gb=35)  # pre-quantized size
        assert s.offload == OffloadMode.NONE

    def test_8b_on_8gb_gpu(self):
        """8B model on RTX 3060 (8GB) — needs FP8 or offload."""
        hw = make_hw(gpu_vram_gb=8, supports_fp8=True)
        s = pick_strategy(hw, model_size_gb=16)
        # FP8 would make it 8.8GB — still too big for 8GB with headroom
        # Should fall back to offload
        assert s.offload != OffloadMode.NONE or s.quantization != QuantMode.NONE


class TestDeviceMapStrategy:
    def test_offload_gives_reasonable_vram(self):
        """Offloaded model should use reasonable GPU memory."""
        hw = make_hw(gpu_vram_gb=24, system_ram_gb=128)
        s = pick_strategy(hw, model_size_gb=80)
        assert s.offload in (OffloadMode.SEQUENTIAL_CPU, OffloadMode.LAYER_HYBRID)
        assert s.estimated_vram_gb <= 24.0

    def test_fits_in_vram_no_offload(self):
        hw = make_hw(gpu_vram_gb=80)
        s = pick_strategy(hw, model_size_gb=40)
        assert s.offload == OffloadMode.NONE
        assert s.compile is True  # should enable compile when fits


class TestImports:
    def test_load_model_importable(self):
        """load_model should be importable from overflowml."""
        import overflowml
        assert callable(overflowml.load_model)

    def test_memory_guard_importable(self):
        import overflowml
        assert callable(overflowml.MemoryGuard)
