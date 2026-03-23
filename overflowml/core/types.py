"""Core data contracts for OverflowML."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class GPUInfo:
    name: str
    total_vram_gb: float
    backend: Literal["cuda", "rocm", "mps", "cpu", "unknown"] = "unknown"
    device_index: int = 0


@dataclass
class HardwareInfo:
    gpus: list[GPUInfo] = field(default_factory=list)
    total_ram_gb: float = 0.0
    torch_version: Optional[str] = None
    torch_cuda_available: bool = False
    torch_cuda_version: Optional[str] = None
    torch_hip_version: Optional[str] = None
    platform: str = ""
    cpu_cores: int = 0
    supports_bf16: bool = False
    supports_fp8: bool = False
    unified_memory: bool = False

    @property
    def total_vram_gb(self) -> float:
        return sum(g.total_vram_gb for g in self.gpus)

    @property
    def primary_backend(self) -> str:
        if self.gpus:
            return self.gpus[0].backend
        return "cpu"

    @property
    def num_gpus(self) -> int:
        return len(self.gpus)


@dataclass
class ModelInfo:
    model_id: str
    task_family: Literal["causal-lm", "diffusers", "embedding", "unknown"] = "unknown"
    architecture: Optional[str] = None
    param_count: Optional[int] = None
    estimated_sizes_gb: dict[str, float] = field(default_factory=dict)
    source: str = "unknown"
    confidence: Literal["low", "medium", "high"] = "low"
    notes: list[str] = field(default_factory=list)


@dataclass
class StrategyCandidate:
    name: str
    viable: bool = True
    estimated_vram_gb: float = 0.0
    estimated_ram_gb: float = 0.0
    estimated_speed: str = "unknown"
    quality_risk: str = "none"
    setup_complexity: str = "simple"
    recommended: bool = False
    rejection_reason: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class PlanResult:
    model: Optional[ModelInfo] = None
    hardware: Optional[HardwareInfo] = None
    recommended: Optional[StrategyCandidate] = None
    strategies: list[StrategyCandidate] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)


@dataclass
class DoctorIssue:
    code: str
    severity: Literal["info", "warn", "error"] = "info"
    message: str = ""
    detected_value: Optional[str] = None
    suggested_fix: Optional[str] = None


@dataclass
class DoctorReport:
    ok: bool = True
    environment: dict = field(default_factory=dict)
    hardware: dict = field(default_factory=dict)
    issues: list[DoctorIssue] = field(default_factory=list)
    fix_commands: list[str] = field(default_factory=list)


@dataclass
class CanRunResult:
    ok: bool = True
    reason: str = ""
    recommended_strategy: Optional[str] = None
    detected_vram_gb: float = 0.0
    detected_ram_gb: float = 0.0
