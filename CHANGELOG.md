# Changelog

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
