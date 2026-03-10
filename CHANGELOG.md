# Changelog

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
