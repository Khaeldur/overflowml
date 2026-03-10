# OverflowML Launch Posts

## r/LocalLLaMA

**Title:** I built OverflowML — run models larger than your GPU with one line of code (pip install overflowml)

**Body:**

I kept hitting the same problem: models too big for my GPU, and every offloading setup required trial and error with different PyTorch flags, quantization configs, and device maps that varied by hardware.

So I built **OverflowML** — it auto-detects your hardware and applies the optimal loading strategy automatically.

**What it does:**

```python
import overflowml

# Auto-detects hardware, picks strategy, loads with optimal device_map
model, tokenizer = overflowml.load_model("meta-llama/Llama-3-70B")
```

That's it. It figures out whether to use FP8 quantization, INT4 (bitsandbytes NF4), CPU offload, sequential offload, or just load directly — based on your actual VRAM, RAM, and GPU capabilities.

**CLI to check what your hardware can run:**

```
$ overflowml benchmark

Model                      Size  Strategy              VRAM   Status
--------------------------------------------------------------------
Llama-3.2-3B                 6G  compile              7.5GB  FAST
Mistral-7B / Llama-3-8B     16G  compile             18.4GB  FAST
Llama-3.1-13B                26G  compile             29.9GB  FAST
Mixtral-8x7B                 93G  sequential cpu       3.0GB  SLOW
Llama-3-70B                 140G  sequential cpu       3.0GB  SLOW
```

**How it works:**

1. **Detect** — identifies GPU (NVIDIA/AMD/Apple Silicon), VRAM, RAM, FP8/BF16 support
2. **Strategy** — decision tree picks optimal approach: direct load > FP8 > model offload > sequential offload > INT4 + offload
3. **Load** — applies the right device_map, quantization config, and memory settings automatically

It also handles known footguns like FP8 being incompatible with CPU offload on Windows, attention_slicing crashing with sequential offload, etc.

**Origin story:** I had a 40GB diffusion model on an RTX 5090 (32GB). Without optimization: 530s/step with VRAM thrashing. After OverflowML: 6.7s/step, 30/30 images in 16 minutes, 3GB peak VRAM.

Works with both HuggingFace transformers and diffusers pipelines. Apple Silicon (unified memory) is detected and handled correctly too.

```
pip install overflowml
pip install overflowml[transformers]  # for load_model()
pip install overflowml[all]           # everything
```

GitHub: https://github.com/Khaeldur/overflowml
PyPI: https://pypi.org/project/overflowml/

MIT licensed. Looking for feedback and edge cases to handle.

---

## r/StableDiffusion

**Title:** OverflowML — automatically run diffusion models larger than your VRAM (40GB model on 24GB GPU, no manual config)

**Body:**

Built this after fighting with FLUX, Qwen-Image-Edit, and other large diffusion models that don't fit in VRAM.

**The problem:** A 40GB model on a 24GB GPU = VRAM thrashing, 530s per step, crashes after 3 images.

**The fix — one line:**

```python
import overflowml

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
overflowml.optimize_pipeline(pipe, model_size_gb=24)
result = pipe("a sunset over mountains")
```

OverflowML auto-detects your GPU and picks the right strategy:
- Model fits? Direct load + torch.compile
- Almost fits? FP8 quantization (halves memory)
- Too big? Sequential CPU offload (1 layer at a time, ~3GB VRAM)

It also handles batch generation with memory cleanup between images:

```python
from overflowml import MemoryGuard

for prompt in prompts:
    with MemoryGuard():
        result = pipe(prompt)
        result.images[0].save("output.png")
```

**Real results (RTX 5090, 40GB model):**
- Before: 530s/step, crashes after 3 images
- After: 6.7s/step, 30/30 images, 16 minutes total, 3GB peak VRAM

Also handles known traps automatically (FP8+offload crash on Windows, attention_slicing+sequential crash, etc).

```
pip install overflowml[diffusers]
```

GitHub: https://github.com/Khaeldur/overflowml | MIT license

---

## HuggingFace Community / Discussion

**Title:** OverflowML: Auto-optimal model loading for any hardware

**Body:**

Sharing a library I built to solve the "model too big for GPU" problem automatically.

**Problem:** Loading large models requires knowing which combination of device_map, quantization, and offloading to use — and it varies by hardware. FP8 doesn't work with CPU offload on Windows. INT4 needs bitsandbytes. Sequential offload and attention_slicing crash together.

**Solution:**

```python
import overflowml

# Detects your hardware, picks strategy, loads with optimal config
model, tokenizer = overflowml.load_model("meta-llama/Llama-3-70B")
```

Under the hood it:
- Detects GPU type, VRAM, RAM, FP8/BF16 support
- Estimates model size from config (no weight download needed)
- Picks the best strategy: direct load, FP8, BitsAndBytes INT4/INT8, model_cpu_offload, or sequential_cpu_offload
- Sets up device_map, max_memory, quantization_config automatically
- Avoids known incompatibilities

Also works with diffusers pipelines:

```python
overflowml.optimize_pipeline(pipe, model_size_gb=40)
```

CLI tool included:

```
$ overflowml benchmark      # shows what models your hardware can run
$ overflowml plan 70        # detailed strategy for a 70GB model
$ overflowml detect         # show hardware capabilities
```

Cross-platform: NVIDIA (CUDA), Apple Silicon (MPS/MLX unified memory), AMD (ROCm planned).

`pip install overflowml[transformers]`

GitHub: https://github.com/Khaeldur/overflowml

---

## Hacker News (Show HN)

**Title:** Show HN: OverflowML – Run AI models larger than your GPU, one line of code

**URL:** https://github.com/Khaeldur/overflowml

**Text (optional comment):**

I built OverflowML after spending too long debugging PyTorch memory errors. The core idea: auto-detect hardware, pick the optimal loading strategy, and apply it — no manual config.

The strategy engine handles the decision tree: if a model fits in VRAM, load directly with torch.compile. If FP8 makes it fit, quantize. If components fit individually, use model_cpu_offload. Otherwise, sequential CPU offload (1 layer at a time, ~3GB VRAM). It also handles known incompatibilities automatically (FP8 + offload on Windows, attention_slicing + sequential, etc).

Works with HuggingFace transformers (`overflowml.load_model()`) and diffusers pipelines (`overflowml.optimize_pipeline()`). Cross-platform: NVIDIA CUDA, Apple Silicon (unified memory), AMD ROCm planned.

Origin: had a 40GB diffusion model on 32GB VRAM. Before: 530s/step with thrashing. After: 6.7s/step, 3GB peak VRAM.

`pip install overflowml` — MIT licensed, looking for feedback.
