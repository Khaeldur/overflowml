"""Live machine state — what is free *right now*, not what the hardware has.

Planning against total VRAM is wrong on a shared box (training jobs, other servers).
Uses nvidia-smi so it sees every process's allocations, works inside WSL, and never initialises CUDA.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

WSL_DRIVERS = Path("/usr/lib/wsl/drivers")
LINUX_PTXJIT = [Path("/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1"),
                Path("/usr/lib64/libnvidia-ptxjitcompiler.so.1")]


@dataclass
class GPUState:
    index: int
    name: str
    total_gb: float
    used_gb: float
    util_pct: float

    @property
    def free_gb(self) -> float:
        return max(0.0, self.total_gb - self.used_gb)


@dataclass
class LiveState:
    gpus: list[GPUState] = field(default_factory=list)
    gpu_processes: list[dict] = field(default_factory=list)
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    cpu_logical: int = 0
    cpu_physical: int = 0
    cpu_busy: float = 0.0          # 0..1, load average / logical cores (clamped)
    is_wsl: bool = False
    wsl_cuda_lib_dir: Optional[str] = None   # set when LD_LIBRARY_PATH must point here (see detect_wsl_cuda_fix)
    source: str = "none"


def _nvidia_smi(args: list[str]) -> Optional[list[list[str]]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, *args, "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return [[c.strip() for c in line.split(",")] for line in out.stdout.strip().splitlines() if line.strip()]


def _num(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return 0.0


def query_gpus() -> tuple[list[GPUState], str]:
    rows = _nvidia_smi(["--query-gpu=index,name,memory.total,memory.used,utilization.gpu"])
    if rows:
        return [GPUState(int(_num(r[0])), r[1], _num(r[2]) / 1024, _num(r[3]) / 1024, _num(r[4]))
                for r in rows if len(r) >= 5], "nvidia-smi"
    try:
        import torch
        if torch.cuda.is_available():
            gpus = []
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                gpus.append(GPUState(i, torch.cuda.get_device_name(i), total / 1024**3,
                                     (total - free) / 1024**3, 0.0))
            return gpus, "torch"
    except Exception:
        pass
    return [], "none"


def query_gpu_processes() -> list[dict]:
    rows = _nvidia_smi(["--query-compute-apps=pid,process_name,used_memory"]) or []
    return [{"pid": int(_num(r[0])), "name": r[1], "used_gb": _num(r[2]) / 1024}
            for r in rows if len(r) >= 3]


def is_wsl() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def detect_wsl_cuda_fix() -> Optional[str]:
    """On WSL2, a Linux NVIDIA userspace package (e.g. libnvidia-compute-*, pulled in by nvidia-cuda-toolkit)
    shadows the Windows driver's libnvidia-ptxjitcompiler; CUDA init then aborts with 'free(): invalid pointer'.
    Returns the driver-store dir to put first on LD_LIBRARY_PATH, or None when no fix is needed."""
    if not is_wsl() or not WSL_DRIVERS.is_dir():
        return None
    if not any(p.exists() for p in LINUX_PTXJIT):
        return None
    dirs = [p.parent for p in WSL_DRIVERS.glob("*/libcuda.so.1.1")
            if (p.parent / "libnvidia-ptxjitcompiler.so.1").exists()]
    if not dirs:
        return None
    best = str(max(dirs, key=lambda d: (d / "libcuda.so.1.1").stat().st_mtime))
    if best in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)[:1]:
        return None
    return best


def cpu_busy_fraction(logical: int) -> float:
    if hasattr(os, "getloadavg"):
        return min(1.0, os.getloadavg()[0] / max(1, logical))
    try:
        import psutil
        return psutil.cpu_percent(interval=0.5) / 100
    except ImportError:
        return 0.0


def live_state() -> LiveState:
    st = LiveState()
    st.gpus, st.source = query_gpus()
    if st.gpus:
        st.gpu_processes = query_gpu_processes()
    try:
        import psutil
        vm = psutil.virtual_memory()
        st.ram_total_gb = vm.total / 1024**3
        st.ram_available_gb = vm.available / 1024**3
        st.cpu_logical = psutil.cpu_count() or 1
        st.cpu_physical = psutil.cpu_count(logical=False) or st.cpu_logical
    except ImportError:
        st.cpu_logical = st.cpu_physical = os.cpu_count() or 1
    st.cpu_busy = cpu_busy_fraction(st.cpu_logical)
    st.is_wsl = is_wsl()
    st.wsl_cuda_lib_dir = detect_wsl_cuda_fix()
    return st
