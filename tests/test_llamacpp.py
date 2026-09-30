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


# ---- GGUF reader + live-budget llama.cpp planner ----

import json as _json
import struct as _struct
import subprocess as _subprocess
import sys as _sys

import pytest as _pytest

from overflowml.core import live as _live
from overflowml.core.llamacpp_plan import plan_gguf
from overflowml.inspect.gguf import read_gguf

MB = 1024**2


def _gguf_str(s):
    b = s.encode()
    return _struct.pack("<Q", len(b)) + b


def write_gguf(path, arch, n_layers, layer_tensors, meta_extra=None):
    """Minimal GGUF v3: metadata + tensor table + zero-filled data. layer_tensors: {suffix: bytes}."""
    meta = {"general.architecture": (8, arch), "general.name": (8, "Test"), "general.alignment": (4, 32),
            f"{arch}.block_count": (4, n_layers), f"{arch}.embedding_length": (4, 1024),
            f"{arch}.attention.head_count": (4, 8), f"{arch}.attention.head_count_kv": (4, 2),
            f"{arch}.context_length": (4, 32768)}
    meta.update(meta_extra or {})
    tensors = [("token_embd.weight", 4 * MB)]
    for b in range(n_layers):
        lt = layer_tensors(b) if callable(layer_tensors) else layer_tensors
        tensors += [(f"blk.{b}.{sfx}", size) for sfx, size in lt.items()]
    tensors += [("output_norm.weight", 32), ("output.weight", 4 * MB)]

    out = bytearray(b"GGUF" + _struct.pack("<IQQ", 3, len(tensors), len(meta)))
    for k, (t, v) in meta.items():
        out += _gguf_str(k) + _struct.pack("<I", t)
        out += _gguf_str(v) if t == 8 else _struct.pack("<I", v)
    off = 0
    for name, size in tensors:
        out += _gguf_str(name) + _struct.pack("<I", 1) + _struct.pack("<Q", size) + _struct.pack("<IQ", 0, off)
        off += (size + 31) // 32 * 32
    out += b"\0" * ((-len(out)) % 32)
    with open(path, "wb") as f:
        f.write(out)
        f.truncate(len(out) + off)
    return path


@_pytest.fixture
def dense_gguf(tmp_path):
    return write_gguf(tmp_path / "dense.gguf", "llama", 10,
                      {"attn_k.weight": 8 * MB, "attn_q.weight": 32 * MB, "ffn_up.weight": 60 * MB})


@_pytest.fixture
def moe_gguf(tmp_path):
    return write_gguf(tmp_path / "moe.gguf", "qwen3moe", 8,
                      {"attn_k.weight": 8 * MB, "ffn_up_exps.weight": 200 * MB, "ffn_down_exps.weight": 200 * MB},
                      {"qwen3moe.expert_count": (4, 64), "qwen3moe.expert_used_count": (4, 4)})


class TestReadGGUF:
    def test_dense_sizes(self, dense_gguf):
        i = read_gguf(dense_gguf)
        assert i.architecture == "llama" and i.n_layers == 10 and not i.is_moe
        assert i.layer_bytes[0] == 100 * MB
        assert i.embed_bytes == 4 * MB
        assert i.output_bytes == 4 * MB + 32
        assert i.n_kv_layers == 10 and i.key_length == 128

    def test_moe_expert_split(self, moe_gguf):
        i = read_gguf(moe_gguf)
        assert i.is_moe and i.n_experts == 64 and i.n_experts_used == 4
        assert i.layer_expert_bytes == [400 * MB] * 8

    def test_hybrid_ssm_layers_have_no_kv(self, tmp_path):
        # Qwen3.5/3.6 layout: linear-attention blocks carry attn_qkv + ssm_*, every 4th block has attn_k
        def layer(b):
            if b % 4 == 3:
                return {"attn_k.weight": MB, "attn_q.weight": MB}
            return {"attn_qkv.weight": MB, "ssm_out.weight": MB}
        i = read_gguf(write_gguf(tmp_path / "hyb.gguf", "qwen35", 8, layer))
        assert i.n_kv_layers == 2

    def test_not_gguf(self, tmp_path):
        p = tmp_path / "x.gguf"
        p.write_bytes(b"NOPE" + b"\0" * 64)
        with _pytest.raises(ValueError):
            read_gguf(p)


class TestPlanGGUF:
    def test_full_gpu_when_budget_large(self, dense_gguf):
        p = plan_gguf(read_gguf(dense_gguf), 24, ctx=4096)
        assert p.mode == "full_gpu" and p.kv_cache_type == "f16"
        assert p.flags[p.flags.index("-ngl") + 1] == "11"
        assert "-t" not in p.flags

    def test_moe_offloads_minimum_expert_layers(self, moe_gguf):
        i = read_gguf(moe_gguf)
        p = plan_gguf(i, 2.5, ctx=4096)
        assert p.mode == "moe_expert_offload"
        assert 0 < p.n_cpu_moe < i.n_layers
        assert p.est_vram_gb <= 2.5
        tighter = plan_gguf(i, 1.5, ctx=4096)
        assert tighter.n_cpu_moe > p.n_cpu_moe
        assert "--n-cpu-moe" in p.flags

    def test_dense_partial_layers(self, dense_gguf):
        p = plan_gguf(read_gguf(dense_gguf), 0.9, ctx=4096)
        assert p.mode == "partial_layers"
        assert 0 < p.n_gpu_layers < 10
        assert any("CPU speed" in w for w in p.warnings)

    def test_cpu_only_when_no_budget(self, dense_gguf):
        p = plan_gguf(read_gguf(dense_gguf), 0.0)
        assert p.mode == "cpu_only" and p.n_gpu_layers == 0
        assert p.flags[p.flags.index("-dev") + 1] == "none"
        assert p.vram_full_gpu_gb > 1.0

    def test_busy_cpu_warns_only_when_offloading(self, moe_gguf):
        i = read_gguf(moe_gguf)
        assert any("busy" in w for w in plan_gguf(i, 1.5, cpu_busy=0.8, cpu_physical=16).warnings)
        assert not any("busy" in w for w in plan_gguf(i, 24, cpu_busy=0.8, cpu_physical=16).warnings)

    def test_threads_scale_with_free_cpu(self, moe_gguf):
        p = plan_gguf(read_gguf(moe_gguf), 1.5, cpu_busy=0.5, cpu_physical=16)
        assert p.flags[p.flags.index("-t") + 1] == "8"

    def test_wsl_env_in_command(self, dense_gguf):
        p = plan_gguf(read_gguf(dense_gguf), 24, wsl_cuda_lib_dir="/usr/lib/wsl/drivers/nv_x")
        assert p.command().startswith("LD_LIBRARY_PATH=/usr/lib/wsl/drivers/nv_x llama-server")


class TestLiveState:
    def test_parses_nvidia_smi(self, monkeypatch):
        monkeypatch.setattr(_live, "_nvidia_smi", lambda a: [["0", "RTX 5090", "32607", "27000", "98"]])
        gpus, src = _live.query_gpus()
        assert src == "nvidia-smi"
        assert round(gpus[0].free_gb, 2) == round((32607 - 27000) / 1024, 2)

    def test_wsl_fix_none_off_wsl(self, monkeypatch):
        monkeypatch.setattr(_live, "is_wsl", lambda: False)
        assert _live.detect_wsl_cuda_fix() is None

    def test_wsl_fix_found(self, monkeypatch, tmp_path):
        drv = tmp_path / "drivers" / "nv_dispi.inf_x"
        drv.mkdir(parents=True)
        (drv / "libcuda.so.1.1").write_bytes(b"")
        (drv / "libnvidia-ptxjitcompiler.so.1").write_bytes(b"")
        linux_lib = tmp_path / "libnvidia-ptxjitcompiler.so.1"
        linux_lib.write_bytes(b"")
        monkeypatch.setattr(_live, "is_wsl", lambda: True)
        monkeypatch.setattr(_live, "WSL_DRIVERS", tmp_path / "drivers")
        monkeypatch.setattr(_live, "LINUX_PTXJIT", [linux_lib])
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        assert _live.detect_wsl_cuda_fix() == str(drv)
        monkeypatch.setenv("LD_LIBRARY_PATH", str(drv))
        assert _live.detect_wsl_cuda_fix() is None


class TestLegacyPlanLlamaCppDetect:
    def test_hw_none_uses_detection(self, monkeypatch):
        import overflowml.strategy as st
        monkeypatch.setattr(st, "detect_hardware", lambda: make_hw(gpu_vram_gb=24))
        assert "-ngl" in plan_llamacpp("m.gguf")["command"]


class TestLlamaCppCLI:
    def test_json_with_budget(self, moe_gguf):
        r = _subprocess.run([_sys.executable, "-m", "overflowml", "llamacpp", str(moe_gguf),
                             "--vram-budget", "2.5", "--ctx", "4096", "--json"],
                            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        out = _json.loads(r.stdout)
        assert out["mode"] == "moe_expert_offload"
        assert out["budget_basis"] == "--vram-budget"
        assert "--n-cpu-moe" in out["command"]

    def test_detect_live_json(self):
        r = _subprocess.run([_sys.executable, "-m", "overflowml", "detect", "--live", "--json"],
                            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        out = _json.loads(r.stdout)
        assert "gpus" in out and "cpu_busy" in out


# --- Split GGUF + hostile-file hardening ---------------------------------------------------

from overflowml.inspect.gguf import GGUFError as _GGUFError


def _gguf_bytes(meta, tensors, version=3):
    """Raw GGUF: meta {key: (type, value)} with u32/string values, tensors [(name, size)]."""
    out = bytearray(b"GGUF" + _struct.pack("<IQQ", version, len(tensors), len(meta)))
    for k, (t, v) in meta.items():
        out += _gguf_str(k) + _struct.pack("<I", t)
        out += _gguf_str(v) if t == 8 else _struct.pack("<I", v)
    off = 0
    for name, size in tensors:
        out += _gguf_str(name) + _struct.pack("<I", 1) + _struct.pack("<Q", size) + _struct.pack("<IQ", 0, off)
        off += (size + 31) // 32 * 32
    out += b"\0" * ((-len(out)) % 32)
    return bytes(out), off


def write_split(tmp_path, n_layers=12, n_shards=3, layer_mb=100):
    """Dense model split llama.cpp-style: shard 1 has the metadata, every shard has a slice of the tensors."""
    paths = []
    per = n_layers // n_shards
    for s in range(n_shards):
        tensors = [("token_embd.weight", 4 * MB)] if s == 0 else []
        tensors += [(f"blk.{b}.ffn_up.weight", layer_mb * MB) for b in range(s * per, (s + 1) * per)]
        if s == n_shards - 1:
            tensors.append(("output.weight", 4 * MB))
        meta = {"split.no": (4, s), "split.count": (4, n_shards)}
        if s == 0:
            meta.update({"general.architecture": (8, "llama"), "general.alignment": (4, 32),
                         "llama.block_count": (4, n_layers), "llama.embedding_length": (4, 1024),
                         "llama.attention.head_count": (4, 8), "llama.attention.head_count_kv": (4, 2)})
        header, data = _gguf_bytes(meta, tensors)
        p = tmp_path / f"model-{s + 1:05d}-of-{n_shards:05d}.gguf"
        with open(p, "wb") as f:
            f.write(header)
            f.truncate(len(header) + data)
        paths.append(p)
    return paths


class TestSplitGGUF:
    def test_shards_are_summed(self, tmp_path):
        paths = write_split(tmp_path)
        i = read_gguf(paths[0])
        assert len(i.shards) == 3
        assert i.layer_bytes == [100 * MB] * 12
        assert i.weights_bytes == 12 * 100 * MB + 8 * MB

    def test_any_shard_reads_the_whole_model(self, tmp_path):
        paths = write_split(tmp_path)
        assert read_gguf(paths[2]).weights_bytes == read_gguf(paths[0]).weights_bytes
        assert read_gguf(paths[2]).n_layers == 12

    def test_missing_shard_fails_closed(self, tmp_path):
        paths = write_split(tmp_path)
        paths[1].unlink()
        with _pytest.raises(_GGUFError, match="missing"):
            read_gguf(paths[0])

    def test_split_model_never_plans_from_first_shard(self, tmp_path):
        # Regression: MiniMax 4 shards (123 GB) planned as full_gpu at 4.4 GB from shard 1 alone
        i = read_gguf(write_split(tmp_path, layer_mb=1000)[0])
        p = plan_gguf(i, 4.0, ctx=4096, ram_available_gb=64)
        assert p.mode != "full_gpu"

    def test_cli_missing_shard_exits_2(self, tmp_path):
        import json as _json
        import subprocess as _sp
        import sys as _sys
        paths = write_split(tmp_path)
        paths[2].unlink()
        r = _sp.run([_sys.executable, "-m", "overflowml", "llamacpp", str(paths[0]), "--vram-budget", "8", "--json"],
                    capture_output=True, text=True)
        assert r.returncode == 2
        assert _json.loads(r.stdout)["error"] == "gguf"


def _hostile(tmp_path, body, name="evil.gguf"):
    p = tmp_path / name
    p.write_bytes(body)
    return p


def _assert_fails_small(path, max_mb=16):
    import tracemalloc
    tracemalloc.start()
    try:
        with _pytest.raises(_GGUFError):
            read_gguf(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < max_mb * MB, f"peak {peak / MB:.0f} MB"


class TestHostileGGUF:
    """PoCs from the red-team review, kept as regressions: each must fail fast without big allocations."""

    def _header(self, n_tensors, n_kv):
        return b"GGUF" + _struct.pack("<IQQ", 3, n_tensors, n_kv)

    def test_block_count_bomb(self, tmp_path):
        # A 106-byte file with block_count=2**30 drove RSS to ~83 GB
        body, _ = _gguf_bytes({"general.architecture": (8, "llama"), "llama.block_count": (4, 2**30)}, [])
        _assert_fails_small(_hostile(tmp_path, body))

    def test_u64_block_count_bomb(self, tmp_path):
        body = self._header(0, 2) + _gguf_str("general.architecture") + _struct.pack("<I", 8) + _gguf_str("llama")
        body += _gguf_str("llama.block_count") + _struct.pack("<IQ", 10, 2**34)
        _assert_fails_small(_hostile(tmp_path, body))

    def test_scalar_array_bomb(self, tmp_path):
        # 49-byte file forced a ~2 GB read before the n<=4096 guard
        body = self._header(0, 1) + _gguf_str("k") + _struct.pack("<IIQ", 9, 10, 2**40)
        _assert_fails_small(_hostile(tmp_path, body))

    def test_string_length_bomb(self, tmp_path):
        body = self._header(0, 1) + _struct.pack("<Q", 2**40)
        _assert_fails_small(_hostile(tmp_path, body))

    def test_deeply_nested_arrays(self, tmp_path):
        body = self._header(0, 1) + _gguf_str("k") + _struct.pack("<I", 9)
        body += _struct.pack("<IQ", 9, 1) * 5000 + _struct.pack("<IQ", 4, 0)
        _assert_fails_small(_hostile(tmp_path, body))

    def test_huge_kv_count(self, tmp_path):
        _assert_fails_small(_hostile(tmp_path, self._header(0, 10**18)))

    def test_huge_tensor_count(self, tmp_path):
        _assert_fails_small(_hostile(tmp_path, self._header(10**18, 0)))

    def test_zero_alignment(self, tmp_path):
        body, _ = _gguf_bytes({"general.architecture": (8, "llama"), "general.alignment": (4, 0)}, [])
        _assert_fails_small(_hostile(tmp_path, body))

    def test_truncated_metadata(self, tmp_path, dense_gguf):
        data = open(dense_gguf, "rb").read()
        _assert_fails_small(_hostile(tmp_path, data[:60], "trunc.gguf"))

    def test_tensor_data_past_end_of_file(self, tmp_path):
        # A partially downloaded shard: the tensor table points beyond the file
        header, _ = _gguf_bytes({"general.architecture": (8, "llama"), "llama.block_count": (4, 1)},
                                [("blk.0.ffn_up.weight", 100 * MB)])
        body = header + b"\0" * 64  # data section cut short
        header2, _ = _gguf_bytes({"general.architecture": (8, "llama"), "llama.block_count": (4, 1)},
                                 [("blk.0.a", 32), ("blk.0.b", 32)])
        # second tensor's offset (32) is past a 16-byte data section
        _assert_fails_small(_hostile(tmp_path, header2 + b"\0" * 16, "cut.gguf"))
        read_gguf(_hostile(tmp_path, body, "ok.gguf"))  # a single tensor to EOF is fine (size = what's there)

    def test_valid_files_still_parse(self, dense_gguf, moe_gguf):
        assert read_gguf(dense_gguf).n_layers == 10
        assert read_gguf(moe_gguf).is_moe
