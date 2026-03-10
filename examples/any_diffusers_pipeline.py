"""Example: Use OverflowML with any diffusers pipeline.

Works with Stable Diffusion, SDXL, Flux, Qwen-Image-Edit, etc.
Just load your pipeline normally, then call optimize_pipeline().
"""

import torch
import overflowml
from overflowml import MemoryGuard

# --- Example 1: SDXL (7GB model, fits most GPUs) ---
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)
strategy = overflowml.optimize_pipeline(pipe)
# On RTX 4090: no offload needed, torch.compile enabled
# On RTX 3060 (12GB): might use FP8 or model offload

result = pipe("a cat in space", num_inference_steps=20)
result.images[0].save("sdxl_output.png")


# --- Example 2: Flux (24GB model, needs offload on most GPUs) ---
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)
strategy = overflowml.optimize_pipeline(pipe, model_size_gb=24)
# On RTX 5090 (32GB): FP8 quantization, fits in VRAM
# On RTX 4090 (24GB): sequential offload, ~3GB VRAM
# On Mac M4 Max (128GB): direct load

result = pipe("a sunset over mountains", num_inference_steps=20)
result.images[0].save("flux_output.png")


# --- Example 3: Batch generation with MemoryGuard ---
guard = MemoryGuard(threshold=0.7)

prompts = ["a cat", "a dog", "a bird", "a fish"]
for i, prompt in enumerate(prompts):
    with guard:  # auto-cleans VRAM between images
        result = pipe(prompt, num_inference_steps=20)
        result.images[0].save(f"batch_{i}.png")
