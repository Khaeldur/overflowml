# OverflowML

[![Tests](https://github.com/Khaeldur/overflowml/actions/workflows/tests.yml/badge.svg)](https://github.com/Khaeldur/overflowml/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/overflowml)](https://pypi.org/project/overflowml/)
[![Python](https://img.shields.io/pypi/pyversions/overflowml)](https://pypi.org/project/overflowml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Run AI models larger than your GPU.** One line of code.

OverflowML auto-detects your hardware (NVIDIA, Apple Silicon, AMD, CPU) and applies the optimal memory strategy to load and run models that don't fit in VRAM. No manual configuration needed.

```python
import overflowml

pipe = load_your_model()  # 40GB model, 24GB GPU? No problem.
overflowml.optimize_pipeline(pipe, model_size_gb=40)
result = pipe(prompt)     # Just works.
```

## The Problem

AI models are getting bigger. A single image generation model can be 40GB+. LLMs regularly hit 70GB-400GB. But most GPUs have 8-24GB of VRAM.

The current solutions are painful:
- **Manual offloading** — you need to know which PyTorch function to call, which flags work together, and which combinations crash
- **Quantization footguns** — FP8 is incompatible with CPU offload on Windows. Attention slicing crashes with sequential offload. INT4 needs specific libraries.
- **Trial and error** — every hardware/model/framework combo has different gotchas

OverflowML handles all of this automatically.

## How It Works

```
Model: 40GB (BF16)          Your GPU: 24GB VRAM
         │                           │
    OverflowML detects mismatch      │
         │                           │
    ┌────▼────────────────────────────▼────┐
    │  Strategy: Sequential CPU Offload    │
    │  Move 1 layer (~1GB) to GPU at a    │
    │  time, compute, move back.          │
    │  Peak VRAM: ~3GB                     │
    │  System RAM used: ~40GB              │
    │  Speed: 33s/image (RTX 5090)        │
    └──────────────────────────────────────┘
```

### Strategy Decision Tree

| Model vs VRAM | Strategy | Peak VRAM | Speed |
|---------------|----------|-----------|-------|
| Model fits with 15% headroom | Direct GPU load | Full | Fastest |
| FP8 model fits | FP8 quantization | ~55% of model | Fast |
| Components fit individually | Model CPU offload | ~70% of model | Medium |
| Nothing fits | Sequential CPU offload | ~3GB | Slower but works |
| Not enough RAM either | INT4 quantization + sequential | ~3GB | Slowest |

### Apple Silicon (Unified Memory)

On Macs, CPU and GPU share the same memory pool — there's nothing to "offload." OverflowML detects this and skips offloading entirely. If the model fits in ~75% of your RAM, it loads directly. If not, quantization is recommended.

| Mac | Unified Memory | Largest Model (4-bit) |
|-----|---------------|----------------------|
| M4 Max | 128GB | ~80B params |
| M3/M4 Ultra | 192GB | ~120B params |
| M3 Ultra | 512GB | 670B params |

## Installation

```bash
pip install overflowml

# With diffusers support:
pip install overflowml[diffusers]

# With quantization:
pip install overflowml[all]
```

Verify installation:

```bash
python -c "import overflowml; print(overflowml.__version__)"
overflowml detect
```

## Usage

### Diffusers Pipeline (Recommended)

```python
import torch
import overflowml
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)

# One line — auto-detects hardware, picks optimal strategy
strategy = overflowml.optimize_pipeline(pipe, model_size_gb=24)
print(strategy.summary())

result = pipe("a sunset over mountains", num_inference_steps=20)
```

### Batch Generation with Memory Guard

```python
from overflowml import MemoryGuard

guard = MemoryGuard(threshold=0.7)  # cleanup at 70% VRAM usage

for prompt in prompts:
    with guard:  # auto-cleans VRAM between iterations
        result = pipe(prompt)
        result.images[0].save(f"output.png")
```

### CLI — Hardware Detection

```bash
$ overflowml detect

=== OverflowML Hardware Detection ===
Accelerator: cuda
GPU: NVIDIA GeForce RTX 5090 (32GB VRAM)
System RAM: 194GB
Overflow capacity: 178GB (total effective: 210GB)
BF16: yes | FP8: yes

$ overflowml plan 40

=== Strategy for 40GB model ===
Offload: sequential_cpu
Dtype: bfloat16
GC cleanup: enabled (threshold 70%)
Estimated peak VRAM: 3.0GB
  → Sequential offload: 1 layer at a time (~3GB VRAM), model lives in 194GB RAM
WARNING: FP8 incompatible with CPU offload on Windows
WARNING: Do NOT enable attention_slicing with sequential offload
```

### Standalone Model

```python
import overflowml

model = load_my_transformer()
strategy = overflowml.optimize_model(model, model_size_gb=14)
```

## Proven Results

Built and battle-tested on a real production pipeline:

| Metric | Before OverflowML | After |
|--------|-------------------|-------|
| Time per step | 530s (VRAM thrashing) | 6.7s |
| Images generated | 0/30 (crashes) | 30/30 |
| Total time | Impossible | 16.4 minutes |
| Peak VRAM | 32GB (thrashing) | 3GB |
| Reliability | Crashes after 3 images | Zero failures |

*40GB model on RTX 5090 (32GB VRAM) + 194GB RAM, sequential offload, Lightning LoRA 4-step*

## MoE Expert Offload — Real Benchmarks

OverflowML's MoE strategy enables running 120B+ parameter models on consumer GPUs by keeping shared layers on GPU and swapping experts from RAM:

| Model | Total Params | Active | VRAM Used | RAM Used | Tokens/s | Strategy |
|-------|-------------|--------|-----------|----------|----------|----------|
| Nemotron 3 Super | 120B | 12B | 29GB (32% GPU) | 63GB (68% CPU) | **5.9 t/s** | Expert offload Q4 |
| Nemotron 3 Nano | 30B | 3.6B | 24GB (100% GPU) | 0GB | **228 t/s** | Full GPU |

*RTX 5090 (32GB VRAM) + 196GB RAM, Ollama, Q4_K_M quantization*

### Optimization Sweep — Finding the Optimal GPU/CPU Split

We tested 12 configurations to find the best strategy for Nemotron 3 Super on a single RTX 5090:

| Config | Context | GPU Layers | CPU/GPU Split | Tokens/s |
|--------|---------|-----------|---------------|----------|
| **Default (auto)** | 32K | auto | 68%/32% | **5.9** |
| Reduced context | 8K | auto | 68%/32% | 5.6 |
| Minimal context | 2K | auto | 68%/32% | 5.8 |
| Fewer GPU layers | 8K | 20 | 77%/23% | 5.3 |
| More GPU layers | 8K | 40 | 54%/46% | 1.3 |
| **All GPU (forced)** | 8K | 99 | 0%/100% | **0.9** |
| 32 threads + batch 1024 | 8K | auto | 68%/32% | 3.0 |

**Key findings:**
- Ollama's auto-detected 68%/32% CPU/GPU split is optimal
- Forcing more onto GPU **hurts** — 100% GPU = 6.5x slower (VRAM thrashing on 86GB model in 32GB)
- Context size barely matters — bottleneck is PCIe expert swapping, not KV cache
- Model can't fit in 32GB at any quantization (minimum GGUF is 52.7GB due to unquantizable Mamba2 SSM state)
- The hard limit is PCIe 5.0 bandwidth (~50 GB/s practical) for expert transfers

```bash
$ overflowml plan 120 --moe 120 12 128 8

=== Strategy for 120GB model ===
MoE: 120B total, 12B active, 128 experts (8 active)
Quantization: int4
Offload: expert_offload
Estimated peak VRAM: 14.8GB
  - MoE expert offload + INT4: 36GB total in RAM, 15GB active on GPU
```

## Known Incompatibilities

These are automatically handled by OverflowML's strategy engine:

| Combination | Issue | OverflowML Fix |
|-------------|-------|----------------|
| FP8 + CPU offload (Windows) | `Float8Tensor` can't move between devices | Skips FP8, uses BF16 |
| `attention_slicing` + sequential offload | CUDA illegal memory access | Never enables both |
| `enable_model_cpu_offload` + 40GB transformer | Transformer exceeds VRAM | Uses sequential offload instead |
| `expandable_segments` on Windows WDDM | Not supported | Gracefully ignored |

## Architecture

```
overflowml/
├── detect.py      — Hardware detection (CUDA, MPS, MLX, ROCm, CPU)
├── strategy.py    — Strategy engine (picks optimal offload + quantization)
├── optimize.py    — Applies strategy to pipelines and models
└── cli.py         — Command-line interface
```

## Cross-Platform Support

| Platform | Accelerator | Status |
|----------|-------------|--------|
| Windows + NVIDIA | CUDA | Production-ready |
| Linux + NVIDIA | CUDA | Production-ready |
| macOS + Apple Silicon | MPS / MLX | Detection ready, optimization in progress |
| Linux + AMD | ROCm | Planned |
| CPU-only | CPU | Fallback always works |

## Troubleshooting

**Hardware not detected / wrong accelerator**
Run `overflowml detect` — if it shows CPU when you have a GPU, install the correct PyTorch build for your platform: [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

**FP8 errors on Windows**
Expected. FP8 is incompatible with CPU offload on Windows — OverflowML automatically falls back to BF16. No action needed.

**OOM despite using sequential offload**
Your system RAM may be insufficient. Try `overflowml plan <size> --no-quantize` to see minimum RAM requirements. INT4 quantization (`pip install overflowml[all]`) reduces the RAM footprint by ~4x.

**`torchao` import error with FP8**
Install: `pip install torchao>=0.5` or use `pip install overflowml[quantize]`.

**`accelerate` not found with `enable_model_cpu_offload`**
Install: `pip install accelerate>=0.30` or use `pip install overflowml[diffusers]`.

**`torch.compile` skipped (Triton not installed)**
Not an error — OverflowML skips compilation silently. On Windows, Triton requires WSL2. On Linux, install with `pip install triton`.

**Slow generation with sequential offload**
Sequential offload is intentionally slow (layers move one-at-a-time). Check `overflowml plan <size>` — if your RAM is large enough, it may suggest a faster strategy like `model_cpu_offload`.

## License

MIT
