"""Example: Run Qwen-Image-Edit-2511 (40GB model) on any GPU.

This model is 40GB in BF16 — too large for most GPUs.
OverflowML auto-detects your hardware and picks the best strategy:
  - RTX 5090 (32GB): sequential offload, ~3GB VRAM, 33s/image
  - RTX 4090 (24GB): sequential offload, ~3GB VRAM, ~50s/image
  - RTX 3090 (24GB): sequential offload, ~3GB VRAM, ~80s/image
  - Mac M2 Ultra (192GB): direct load, full speed
  - Mac M4 Max (128GB): direct load, full speed
"""

import torch
from pathlib import Path
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

import overflowml
from overflowml import MemoryGuard

# 1. Detect hardware
hw = overflowml.detect_hardware()
print(hw.summary())

# 2. Load model
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

# 3. Optimize — ONE LINE does everything
strategy = overflowml.optimize_pipeline(pipe, model_size_gb=40)
print(strategy.summary())

# 4. Generate with memory guard (prevents VRAM fragmentation)
face = Image.open("avatar.png").convert("RGB")
guard = MemoryGuard(threshold=0.7, verbose=True)

prompts = [
    "Photograph of this woman, clean white studio, confident gaze, editorial headshot.",
    "Photograph of this woman, outdoor golden hour, casual white linen shirt, natural smile.",
    "Photograph of this woman, cozy cafe, holding ceramic coffee cup, relaxed expression.",
]

for i, prompt in enumerate(prompts):
    with guard:
        result = pipe(
            image=[face],
            prompt=f"[image 1] is the reference face. {prompt}",
            negative_prompt="low quality, blurry, distorted",
            true_cfg_scale=4.0,
            guidance_scale=1.0,
            num_inference_steps=25,
            width=832,
            height=1040,
            generator=torch.manual_seed(42),
        )
        result.images[0].save(f"output_{i+1}.png")
        print(f"Image {i+1} saved")
