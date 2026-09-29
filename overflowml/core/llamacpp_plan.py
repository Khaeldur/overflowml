"""llama.cpp launch planner for GGUF models, sized against a VRAM *budget* (usually live free VRAM).

Candidate order (first that fits wins):
  1. everything on GPU, f16 KV
  2. everything on GPU, q8_0 KV (halves context memory, negligible quality cost)
  3. MoE: keep the first N layers' experts in RAM (--n-cpu-moe N), smallest N that fits
     dense: offload as many repeating layers as fit (-ngl N)
  4. CPU only
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..inspect.gguf import GGUFInfo

GB = 1024**3
OVERHEAD_GB = 0.5       # CUDA context + compute buffers at default ubatch (estimate; ~0.3-0.6 measured)
BUSY_CPU = 0.5          # above this load fraction, CPU-side weights compete with other jobs
LOW_GPU_SHARE = 0.5     # below this, an offloaded plan runs at roughly CPU speed


@dataclass
class LlamaCppPlan:
    fits: bool
    mode: str                       # full_gpu | moe_expert_offload | partial_layers | cpu_only
    flags: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    kv_cache_type: str = "f16"
    n_gpu_layers: int = 0
    n_cpu_moe: int = 0
    est_vram_gb: float = 0.0
    est_ram_gb: float = 0.0
    vram_budget_gb: float = 0.0
    vram_full_gpu_gb: float = 0.0              # VRAM the whole model needs (q8_0 KV) — wait for this much
    gpu_share_of_active_weights: float = 1.0   # 1.0 = every per-token weight read is from VRAM
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def command(self, binary: str = "llama-server") -> str:
        env = " ".join(f"{k}={v}" for k, v in self.env.items())
        return (env + " " if env else "") + " ".join([binary, *self.flags])


def _active_bytes(info: GGUFInfo, expert_bytes: float) -> float:
    """Bytes read per token: dense weights + the used fraction of expert weights."""
    if not info.is_moe:
        return expert_bytes
    return expert_bytes * info.n_experts_used / info.n_experts


def plan_gguf(
    info: GGUFInfo,
    vram_budget_gb: float,
    *,
    ctx: int = 8192,
    ram_available_gb: float = 0.0,
    cpu_busy: float = 0.0,
    cpu_physical: int = 0,
    wsl_cuda_lib_dir: Optional[str] = None,
) -> LlamaCppPlan:
    budget = vram_budget_gb * GB
    overhead = OVERHEAD_GB * GB
    n = info.n_layers
    experts = info.layer_expert_bytes
    total_experts = sum(experts)
    gpu_full = sum(info.layer_bytes) + info.output_bytes   # token_embd stays on CPU in llama.cpp

    def kv(t):
        return info.kv_bytes_per_token(t) * ctx

    plan = None
    for t in ("f16", "q8_0"):
        need = gpu_full + kv(t) + overhead
        if need <= budget:
            plan = LlamaCppPlan(True, "full_gpu", kv_cache_type=t, n_gpu_layers=n + 1,
                                est_vram_gb=need / GB, est_ram_gb=info.embed_bytes / GB)
            break

    if plan is None and info.is_moe:
        base = gpu_full + kv("q8_0") + overhead
        moved = 0
        for k in range(1, n + 1):
            moved += experts[k - 1]
            if base - moved <= budget:
                gpu_exp = total_experts - moved
                dense = gpu_full - total_experts
                on_gpu = _active_bytes(info, gpu_exp) + dense
                all_active = _active_bytes(info, total_experts) + dense
                plan = LlamaCppPlan(True, "moe_expert_offload", kv_cache_type="q8_0", n_gpu_layers=n + 1,
                                    n_cpu_moe=k, est_vram_gb=(base - moved) / GB,
                                    est_ram_gb=(moved + info.embed_bytes) / GB,
                                    gpu_share_of_active_weights=on_gpu / all_active)
                break

    if plan is None and not info.is_moe:
        layer = info.layer_bytes
        for k in range(n, 0, -1):
            need = sum(layer[n - k:]) + kv("q8_0") * k / n + overhead
            if need <= budget:
                gpu_share = sum(layer[n - k:]) / max(1, gpu_full)
                plan = LlamaCppPlan(True, "partial_layers", kv_cache_type="q8_0", n_gpu_layers=k,
                                    est_vram_gb=need / GB,
                                    est_ram_gb=(info.weights_bytes - sum(layer[n - k:])) / GB,
                                    gpu_share_of_active_weights=gpu_share)
                break

    if plan is None:
        plan = LlamaCppPlan(True, "cpu_only", kv_cache_type="q8_0", n_gpu_layers=0,
                            est_ram_gb=(info.weights_bytes + kv("q8_0")) / GB, gpu_share_of_active_weights=0.0)
        plan.warnings.append("Not even the attention/shared layers fit the VRAM budget — CPU-only inference")

    plan.vram_budget_gb = vram_budget_gb
    plan.vram_full_gpu_gb = (gpu_full + kv("q8_0") + overhead) / GB
    plan.flags = ["-m", info.path, "-c", str(ctx), "-ngl", str(plan.n_gpu_layers), "-fa", "on"]
    if plan.kv_cache_type != "f16":
        plan.flags += ["-ctk", plan.kv_cache_type, "-ctv", plan.kv_cache_type]
    if plan.n_cpu_moe:
        plan.flags += ["--n-cpu-moe", str(plan.n_cpu_moe)]
    if plan.mode == "cpu_only":
        # -ngl 0 alone still creates a CUDA context + compute buffers (~1GB measured) on a GPU with no room
        plan.flags += ["-dev", "none"]

    on_cpu = plan.mode != "full_gpu"
    if on_cpu and cpu_physical:
        threads = max(4, int(cpu_physical * (1 - cpu_busy)))
        plan.flags += ["-t", str(threads)]

    if wsl_cuda_lib_dir:
        plan.env["LD_LIBRARY_PATH"] = wsl_cuda_lib_dir
        plan.notes.append("WSL: a Linux NVIDIA package shadows the Windows driver's ptxjitcompiler — "
                          "without LD_LIBRARY_PATH llama.cpp aborts with 'free(): invalid pointer'")

    if on_cpu and ram_available_gb and plan.est_ram_gb > ram_available_gb * 0.9:
        plan.warnings.append(f"CPU-side weights ({plan.est_ram_gb:.1f}GB) close to available RAM "
                             f"({ram_available_gb:.1f}GB) — expect paging")
    if on_cpu and cpu_busy >= BUSY_CPU:
        plan.warnings.append(
            f"CPU is {cpu_busy:.0%} busy with other work and {1 - plan.gpu_share_of_active_weights:.0%} of "
            "per-token weight reads happen on CPU — generation will be far slower than the idle-machine "
            "rate; prefer waiting for VRAM or pausing CPU-heavy jobs")

    if on_cpu and plan.gpu_share_of_active_weights < LOW_GPU_SHARE:
        plan.warnings.append(
            f"Only {plan.gpu_share_of_active_weights:.0%} of per-token weights on GPU — this runs at roughly "
            f"CPU speed; full-GPU speed needs {plan.vram_full_gpu_gb:.1f}GB free VRAM")

    if plan.mode == "full_gpu":
        plan.notes.append(f"Whole model on GPU, {plan.kv_cache_type} KV cache")
    elif plan.mode == "moe_expert_offload":
        plan.notes.append(f"Experts of the first {plan.n_cpu_moe}/{n} layers stay in RAM; attention, shared "
                          f"experts and the rest on GPU ({plan.gpu_share_of_active_weights:.0%} of active "
                          "weights read from VRAM)")
    elif plan.mode == "partial_layers":
        plan.notes.append(f"{plan.n_gpu_layers}/{n} layers on GPU "
                          f"({plan.gpu_share_of_active_weights:.0%} of weights read from VRAM)")
    plan.notes.append(f"Estimates: VRAM {plan.est_vram_gb:.1f}GB of {vram_budget_gb:.1f}GB budget "
                      f"(incl. ~{OVERHEAD_GB:.1f}GB CUDA/compute overhead), KV {plan.kv_cache_type} "
                      f"{kv(plan.kv_cache_type) / GB:.2f}GB @ {ctx} ctx")
    return plan
