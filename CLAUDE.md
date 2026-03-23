# OverflowML — Claude Code Project Config

## Overview
Auto-detect hardware + apply optimal memory strategy for AI models that don't fit in GPU VRAM.
PyPI: `overflowml` | GitHub: `Khaeldur/overflowml` | MIT license

## Architecture
```
overflowml/
├── __init__.py         — Public API: inspect_model, plan, doctor + legacy exports
├── cli.py              — CLI: detect, inspect, plan, doctor, benchmark, load
├── __main__.py         — python -m overflowml entry point
├── core/
│   ├── types.py        — Data contracts: ModelInfo, HardwareInfo, StrategyCandidate, PlanResult, etc.
│   ├── hardware.py     — New hardware detection → HardwareInfo
│   ├── planner.py      — plan(), compare_strategies() → PlanResult
│   └── explain.py      — Gotcha-aware reasoning builder
├── inspect/
│   ├── hf_probe.py     — HF Hub metadata probing (no weight downloads)
│   ├── model_estimator.py — inspect_model() → ModelInfo
│   └── arch_registry.py — Architecture classification + param estimation
├── doctor/
│   ├── checks.py       — Individual health checks (torch, GPU, deps, driver, fit)
│   └── report.py       — run() → DoctorReport
├── detect.py           — Legacy shim → core.hardware
├── strategy.py         — Strategy engine (unchanged, wrapped by planner)
├── optimize.py         — Applies strategy to pipelines/models
└── transformers_ext.py — HuggingFace transformers integration
tests/ (148 tests)
├── test_types.py       — Data contract tests
├── test_hardware_new.py — New hardware detection tests
├── test_inspect.py     — Model inspection + estimation tests
├── test_planner.py     — Planner, compare, explanation tests
├── test_doctor.py      — Doctor checks + CLI tests
├── test_cli.py         — All CLI subprocess tests
├── test_strategy.py    — Strategy decision tree tests
├── test_multi_gpu.py   — Multi-GPU distribution tests
├── test_transformers.py — Transformers integration tests
├── test_detect.py      — ROCm detection tests
├── test_optimize.py    — Device placement, MemoryGuard tests
└── test_llamacpp.py    — plan_llamacpp + _max_memory_map tests
```

## Key Commands
```bash
# Run tests
cd C:/Users/mvait/Desktop/overflowml && python -m pytest tests/ -v

# Build package
python -m build

# Publish to PyPI (token stored in keyring)
python -m twine upload dist/*

# Test CLI
python -m overflowml detect
python -m overflowml benchmark
python -m overflowml plan 40
```

## Version Bumping
Update version in TWO places:
1. `overflowml/__init__.py` — `__version__ = "X.Y.Z"`
2. `pyproject.toml` — `version = "X.Y.Z"`

## Publishing Workflow
1. Bump version in both files
2. Run tests: `python -m pytest tests/ -v`
3. Build: `python -m build`
4. Upload: `python -m twine upload dist/overflowml-X.Y.Z*`
5. Commit + push to GitHub
6. Create GitHub release: `gh release create vX.Y.Z --title "vX.Y.Z" --notes "..."`

## Strategy Decision Tree (strategy.py)
Priority order for discrete GPUs:
1. Direct load (model + 15% headroom fits VRAM)
2. FP8 quantization (55% of model fits VRAM)
3. FP8 + model_cpu_offload (components fit individually)
4. Model CPU offload (no quantization)
5. Sequential CPU offload (~3GB VRAM, model in RAM)
6. INT4 + sequential (not enough RAM for full model)
7. Disk offload (last resort)

Apple Silicon: unified memory, no offloading needed if model < 75% RAM.

## Known Gotchas
- FP8 + CPU offload crashes on Windows (Float8Tensor device mismatch)
- attention_slicing + sequential offload = CUDA illegal memory access
- expandable_segments not supported on Windows WDDM
- Version must be updated in BOTH __init__.py AND pyproject.toml

## Dependencies
Core: torch, psutil
Optional: transformers, accelerate, diffusers, torchao, bitsandbytes, mlx
Dev: pytest, ruff
