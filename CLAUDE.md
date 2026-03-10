# OverflowML — Claude Code Project Config

## Overview
Auto-detect hardware + apply optimal memory strategy for AI models that don't fit in GPU VRAM.
PyPI: `overflowml` | GitHub: `Khaeldur/overflowml` | MIT license

## Architecture
```
overflowml/
├── detect.py          — Hardware detection (CUDA, MPS, MLX, ROCm, CPU)
├── strategy.py        — Strategy engine (picks offload + quantization)
├── optimize.py        — Applies strategy to diffusers pipelines and models
├── transformers_ext.py — HuggingFace transformers integration (load_model)
├── cli.py             — CLI: detect, plan, benchmark, load
└── __init__.py        — Public API exports
tests/
├── test_strategy.py   — Strategy decision tree tests (20 tests)
└── test_transformers.py — Transformers integration tests (5 tests)
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
