# Nemotron 3 Super Benchmark Posts — Ready to Submit

## 1. HuggingFace Discussion
**URL:** https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF/discussions → New Discussion

**Title:** RTX 5090 (32GB) Benchmark: Nemotron 3 Super Q4_K_M via Ollama — 5.9 tokens/s with expert offload

**Body:**

Ran a full optimization sweep on Nemotron 3 Super (Q4_K_M) using a single RTX 5090 (32GB VRAM) + 196GB RAM.

**Setup:** Ollama, Windows 11, Q4_K_M quantization

**Best result:** 5.9 tokens/s at 68% CPU / 32% GPU split (auto-detected by Ollama)

**Key findings from 12 configurations tested:**
- Ollama's auto-detected 68%/32% CPU/GPU split is optimal
- Forcing 100% GPU = 0.9 t/s (6.5x slower due to VRAM thrashing — 86GB model in 32GB)
- Context size (2K-32K) barely affects throughput — bottleneck is PCIe expert swapping
- Model can't fit in 32GB at any quant level (minimum GGUF is 52.7GB)
- TTFT: ~1.1s warm

| Config | CPU/GPU Split | Tokens/s |
|--------|--------------|----------|
| Auto (optimal) | 68%/32% | 5.9 |
| 40 GPU layers | 54%/46% | 1.3 |
| All GPU (forced) | 0%/100% | 0.9 |

For comparison, Nemotron 3 Nano on the same hardware: 228 t/s (fits fully in GPU).

Full benchmark data and MoE strategy analysis: https://github.com/Khaeldur/overflowml

Hardware: RTX 5090 32GB, 196GB DDR5, PCIe 5.0 x16, Windows 11 + Ollama

---

## 2. Hacker News (Show HN)
**URL:** https://news.ycombinator.com/submit

**Title:** Show HN: OverflowML – Run 120B models on a single GPU with MoE expert offload

**URL field:** https://github.com/Khaeldur/overflowml

**Text:**

I built OverflowML to auto-detect hardware and pick the optimal memory strategy for running AI models that don't fit in VRAM.

Just benchmarked NVIDIA's new Nemotron 3 Super (120B params, 12B active) on a single RTX 5090 (32GB VRAM):

- 5.9 tokens/s with expert offload (shared layers on GPU, experts swap from 196GB RAM)
- Tested 12 configurations — Ollama's auto-detected 68/32 CPU/GPU split was optimal
- Forcing 100% GPU = 0.9 t/s (VRAM thrashing)
- The hard limit is PCIe 5.0 bandwidth for expert transfers

For comparison, the smaller Nemotron 3 Nano (30B, fits in GPU): 228 t/s.

`pip install overflowml` — one command to detect hardware and get the optimal loading strategy for any model size.

---

## 3. NVIDIA Developer Forums
**URL:** https://forums.developer.nvidia.com/t/nemotron-3-nano-30b-with-llama-cpp-playbook/355147 → Reply

**Body:**

Sharing benchmark results for Nemotron 3 Super on a single RTX 5090 (32GB VRAM) + 196GB DDR5 RAM:

- Runtime: Ollama (Q4_K_M quantization)
- Best throughput: 5.9 tokens/s
- Memory split: 68% CPU / 32% GPU (expert offload)
- TTFT: ~1.1s (warm)

Tested 12 configurations including forced GPU layers, context sizes, thread counts, and batch sizes. The auto-detected 68/32 split was optimal — forcing more onto GPU caused VRAM thrashing (0.9 t/s at 100% GPU).

Note: llama.cpp CUDA builds currently segfault on WSL2 with CUDA toolkit 13.1 + driver 595.79 (CUDA 13.2). The segfault occurs during cuInit() static initialization — appears to be a toolkit/driver ABI mismatch. Ollama works because it bundles its own CUDA runtime.

Also noting that Nemotron 3 Super support was merged into upstream llama.cpp (#20411) so once the toolkit mismatch is resolved, native llama-server should work.

Full optimization data: https://github.com/Khaeldur/overflowml
