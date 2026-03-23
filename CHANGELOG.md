# Changelog

## [0.6.1] - 2026-03-23

### Fixed
- **Explanation contract**: Every plan output now references at least one trap checked, even on Linux/macOS direct-load (was showing zero traps)
- **`can_run()` RAM check**: Models too large even with INT4 for available RAM now correctly return `ok=False`

### Added
- **`can_run()` function**: `from overflowml import can_run` — CI/CD gating API with `max_offload` parameter
- **`overflowml can-run` CLI**: `overflowml can-run 40 --json` exits 1 if model can't run
- **`huggingface_hub` optional dep**: `pip install overflowml[hub]` for model inspection without full transformers
- **Deprecation warnings**: `overflowml.detect` and `overflowml.optimize` emit `DeprecationWarning` on direct import (suppressed when using top-level API)
- 18 new tests (166 total)

### Changed
- README Architecture section updated to show new `core/`/`inspect/`/`doctor/` structure
- Removed empty placeholder packages (`monitor/`, `integrations/`, `benchmark/`, `utils/`)

## [0.6.0] - 2026-03-23

### Added
- **New package structure**: `core/`, `inspect/`, `doctor/` subpackages with backward-compatible legacy shims
- **`overflowml doctor`**: Environment health checker — detects CPU-only torch on GPU machines, missing deps, driver mismatches, model fit issues. Actionable fix commands for every problem. `--json` output for CI.
- **`overflowml inspect <model_id>`**: Estimate model size from HuggingFace Hub without downloading weights. Uses safetensors index (exact) or config.json architecture estimation. Shows fp16/int8/int4 sizes + confidence.
- **`overflowml plan <model_or_size>`**: Now accepts HuggingFace model IDs (auto-detects size) or numeric GB. `--assume-size-gb` for manual override.
- **`--compare` mode**: Shows all viable strategies side-by-side with speed/VRAM/quality tradeoffs. Rejected strategies shown with reasons.
- **`--json` on plan/inspect/doctor**: Machine-readable output for all major commands.
- **Reasoning section**: Every plan output explains what was tried, what was rejected, and why — referencing specific known traps (FP8+offload on Windows, attention_slicing+sequential, expandable_segments on WDDM).
- **New data contracts**: `ModelInfo`, `HardwareInfo`, `GPUInfo`, `StrategyCandidate`, `PlanResult`, `DoctorIssue`, `DoctorReport`, `CanRunResult`
- **New public API**: `inspect_model()`, `plan()`, `doctor.run()` alongside legacy `detect_hardware()`, `pick_strategy()`, `optimize_pipeline()`
- 56 new tests (148 total)

### Changed
- `Strategy.summary()` now accepts `include_notes` parameter for separating config from reasoning
- Enriched notes on all `pick_strategy()` return paths (INT4, disk offload)

## [0.5.1] - 2026-03-23

### Security
- `trust-remote-code` warning now prints to stderr (not lost in stdout redirects)
- Disk offload path uses `~/.cache/overflowml/offload/` instead of hardcoded relative `offload_cache` in CWD; configurable via `OVERFLOWML_OFFLOAD_DIR` env var

### Fixed
- **NameError crash**: `optimize.py` `_apply_strategy()` referenced undefined `device` variable when verbose=True and no offload — would crash at runtime
- **ZeroDivisionError**: MemoryGuard could divide by zero if GPU reported 0 total memory
- **IndexError**: `_max_memory_map()` crashed if `gpu_vram_gbs` list was shorter than `num_gpus`
- **HardwareProfile**: `__post_init__` now auto-pads `gpu_vram_gbs` if mismatched with `num_gpus`
- `_max_memory_map()` no longer allocates 8GB CPU when system has 0 RAM

### Added
- CLI input validation: MoE params reject `active > total`, benchmark `--custom` rejects zero/negative sizes
- 7 new edge case tests (92 total): zero VRAM GPU, MoE validation, mismatched GPU list, zero RAM

## [0.5.0] - 2026-03-23

### Fixed
- **ROCm device placement bugs**: AMD GPUs were silently placed on CPU instead of CUDA (via HIP). Fixed in `optimize.py` (2 locations) and `transformers_ext.py` (`_max_memory_map`)
- **MemoryGuard CUDA-only**: Now supports ROCm (via HIP) and Apple MPS — `torch.mps.empty_cache()` and `torch.mps.synchronize()` used on Apple Silicon

### Added
- `__main__.py` — `python -m overflowml` now works
- CLI `--help` shows usage examples
- `_pick_device()` helper centralizes device selection (CUDA, ROCm, MPS, CPU)
- 33 new tests (85 total): CLI subprocess tests, MemoryGuard, param estimation, pipeline detection, `plan_llamacpp`, `_max_memory_map`, device placement

## [0.4.2] - 2026-03-23

### Added
- AMD ROCm detection: GPUs on ROCm builds of PyTorch now correctly identified as `Accelerator.ROCm` instead of falling back to CPU
- ROCm correctly sets `supports_fp8=False` (torchao FP8 is CUDA-only)
- 8 new tests: 4 strategy tests for ROCm, 4 detection-level tests (CUDA vs ROCm disambiguation, no-GPU path)

## [0.4.1] - 2026-03-23

### Changed
- PyPI metadata: Beta status, Python 3.10–3.13 classifiers, OS classifiers, expanded keywords
- Added project URLs (Repository, Changelog, Bug Tracker) to PyPI listing
- Added CI badge, install verification, and Troubleshooting section to README
- Added GitHub CI workflow, issue templates, and SECURITY.md

## [0.4.0] - 2026-03-23

### Added
- MoE registry (`MOE_REGISTRY`) with pre-configured profiles for popular MoE models
- `get_moe_profile()` helper for registry lookups
- `plan_llamacpp()` for generating llama.cpp launch flags from a strategy
- `overflowml plan --moe` CLI flag for MoE-aware strategy planning
- MoE expert offload strategy tier: keeps shared layers on GPU, swaps experts from RAM
- llama.cpp launch flags in `plan` output
- 19 new multi-GPU tests (44 total)

### Changed
- `benchmark` output now includes GPUs column and MULTI-GPU status
- Strategy decision tree extended with multi-GPU distribution tiers

## [0.3.0] - 2026-03-10

### Added
- Multi-GPU detection and distribution via `device_map="auto"` (accelerate)
- `DistributionMode` enum: `NONE`, `DEVICE_MAP_AUTO`
- Multi-GPU strategy tier: distribute → FP8 + distribute → distribute + CPU offload
- `num_gpus`, `gpu_names`, `gpu_vram_gbs`, `total_gpu_vram_gb` fields on HardwareProfile
- `distribution` and `num_gpus_used` fields on Strategy
- Expanded model benchmark: 25 models including Nemotron-4 340B, NVLM-D-72B, VILA, Parakeet, Canary
- GPUs column and MULTI-GPU status in benchmark output
- 19 new multi-GPU tests (44 total)

### Security
- Added input validation: `model_size` rejects negative/zero values
- Added `--trust-remote-code` warning in CLI
- Replaced `__import__("subprocess")` with standard import
- Added upper-bound version constraints on core dependencies

### Changed
- `effective_memory_gb` now uses `total_gpu_vram_gb` for multi-GPU systems
- `_max_memory_map()` loops through all GPUs for per-GPU memory allocation
- `_detect_cuda()` detects all GPUs, uses smallest VRAM for conservative allocation

## [0.2.1] - 2026-03-10

### Added
- `overflowml benchmark` CLI command — shows what models your hardware can run
- Popular models table: Llama, Mistral, Mixtral, SDXL, FLUX, Qwen
- `--custom` flag for testing custom model sizes

## [0.2.0] - 2026-03-10

### Added
- HuggingFace transformers integration (`overflowml.load_model()`)
- Auto model size estimation from HuggingFace config (no weight download needed)
- `overflowml load <model>` CLI command with `--chat` mode
- BitsAndBytes INT4/INT8 quantization support
- `device_map` and `max_memory` auto-configuration
- 25 tests (20 strategy + 5 transformers)

## [0.1.0] - 2026-03-10

### Added
- Hardware detection: CUDA, MPS, MLX, ROCm, CPU
- Strategy engine with decision tree for optimal loading
- `optimize_pipeline()` for diffusers pipelines
- `optimize_model()` for standalone models
- `MemoryGuard` context manager for batch generation
- FP8, INT8, INT4 quantization support
- Sequential CPU offload, model CPU offload, disk offload
- CLI: `overflowml detect`, `overflowml plan`
- Cross-platform: Windows, Linux, macOS (Apple Silicon)
