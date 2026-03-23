"""Tests for plan_llamacpp and _max_memory_map."""

from overflowml.detect import Accelerator, HardwareProfile
from overflowml.strategy import MoEProfile, plan_llamacpp
from overflowml.transformers_ext import _max_memory_map


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


class TestPlanLlamaCpp:
    def test_dense_model(self):
        hw = make_hw(gpu_vram_gb=24)
        result = plan_llamacpp("model.gguf", hw=hw)
        assert "command" in result
        assert "flags" in result
        assert "notes" in result
        assert "-m model.gguf" in result["command"]
        assert "-ngl" in result["command"]

    def test_moe_model(self):
        moe = MoEProfile(
            total_params_b=120, active_params_b=12,
            num_experts=128, num_active_experts=8,
            shared_layers_gb=36, expert_size_gb=84,
        )
        hw = make_hw(gpu_vram_gb=32, system_ram_gb=128)
        result = plan_llamacpp("moe.gguf", moe=moe, hw=hw)
        assert "--mlock" in result["command"]
        assert any("MoE" in n for n in result["notes"])

    def test_custom_port_and_context(self):
        hw = make_hw()
        result = plan_llamacpp("m.gguf", hw=hw, context_size=4096, port=9090)
        assert "-c 4096" in result["command"]
        assert "--port 9090" in result["command"]


class TestMaxMemoryMap:
    def test_cuda_single_gpu(self):
        hw = make_hw(gpu_vram_gb=24, gpu_vram_gbs=[24.0])
        mem = _max_memory_map(hw)
        assert 0 in mem
        assert "cpu" in mem
        assert "20GiB" in mem[0]

    def test_cuda_multi_gpu(self):
        hw = make_hw(
            gpu_vram_gb=24, num_gpus=2,
            gpu_vram_gbs=[24.0, 24.0],
            total_gpu_vram_gb=48.0,
        )
        mem = _max_memory_map(hw)
        assert 0 in mem
        assert 1 in mem
        assert "cpu" in mem

    def test_rocm_gets_gpu_allocation(self):
        hw = make_hw(
            accelerator=Accelerator.ROCm,
            gpu_vram_gb=16, gpu_vram_gbs=[16.0],
        )
        mem = _max_memory_map(hw)
        assert 0 in mem  # ROCm should get GPU allocation just like CUDA

    def test_cpu_only(self):
        hw = make_hw(accelerator=Accelerator.CPU, gpu_vram_gb=0, gpu_vram_gbs=[])
        mem = _max_memory_map(hw)
        assert 0 not in mem  # no GPU
        assert "cpu" in mem

    def test_custom_reserve(self):
        hw = make_hw(gpu_vram_gb=24, gpu_vram_gbs=[24.0])
        mem = _max_memory_map(hw, reserve_gpu_gb=8)
        assert "16GiB" in mem[0]

    def test_mismatched_gpu_list_no_crash(self):
        hw = make_hw(gpu_vram_gb=24, num_gpus=4, gpu_vram_gbs=[24.0], total_gpu_vram_gb=96.0)
        mem = _max_memory_map(hw)
        assert 0 in mem
        assert 3 in mem  # should still produce entries for all 4 GPUs

    def test_zero_ram(self):
        hw = make_hw(gpu_vram_gb=24, gpu_vram_gbs=[24.0], system_ram_gb=0)
        mem = _max_memory_map(hw)
        assert "cpu" not in mem  # no CPU allocation when 0 RAM
