"""Example: Load any LLM with optimal memory strategy.

OverflowML auto-detects your hardware and picks the best way to load:
  - RTX 4090 (24GB) + 64GB RAM + Llama-3-70B (140GB): sequential offload, ~3GB VRAM
  - RTX 5090 (32GB) + 194GB RAM + Qwen2-72B (144GB): sequential offload, ~3GB VRAM
  - Mac M4 Max (128GB) + Llama-3-70B: INT4 quantization, fits in unified memory
  - RTX 3060 (12GB) + Llama-3-8B (16GB): FP8 quantization, fits in VRAM
"""

import overflowml

# === One-liner: auto-everything ===
model, tokenizer = overflowml.load_model("meta-llama/Llama-3.1-8B-Instruct")

# Generate
inputs = tokenizer("The future of AI is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# === With options ===
model, tokenizer = overflowml.load_model(
    "Qwen/Qwen2-72B-Instruct",
    model_size_gb=144,          # override auto-estimate
    prefer_speed=True,          # prioritize speed over VRAM savings
    trust_remote_code=True,
)


# === CLI equivalent ===
# overflowml load meta-llama/Llama-3.1-8B-Instruct --chat
# overflowml load Qwen/Qwen2-72B-Instruct --size 144 --chat
