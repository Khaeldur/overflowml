"""CLI tool for hardware detection and strategy recommendation."""

import argparse
import logging
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from .detect import detect_hardware
from .strategy import DistributionMode, MoEProfile, OffloadMode, pick_strategy, plan_llamacpp


def main():
    parser = argparse.ArgumentParser(
        prog="overflowml",
        description="OverflowML — Run AI models larger than your GPU",
        epilog="Examples:\n"
               "  overflowml detect\n"
               "  overflowml plan 40\n"
               "  overflowml plan 120 --moe 120 12 128 8\n"
               "  overflowml benchmark --custom 70 140\n"
               "  overflowml load meta-llama/Llama-3-8B --chat\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # --- detect
    sub.add_parser("detect", help="Detect hardware and show capabilities")

    # --- plan
    plan = sub.add_parser("plan", help="Plan optimal loading strategy for a model")
    plan.add_argument("model_size", type=float, help="Model size in GB (BF16 weights, must be > 0)")
    plan.add_argument("--fast", action="store_true", help="Prefer speed over VRAM savings")
    plan.add_argument("--no-quantize", action="store_true", help="Disable quantization")
    plan.add_argument("--moe", nargs=4, metavar=("TOTAL_B", "ACTIVE_B", "EXPERTS", "ACTIVE_EXPERTS"),
                       help="MoE config: total_params_B active_params_B num_experts active_experts")

    # --- benchmark
    bench = sub.add_parser("benchmark", help="Show what models your hardware can run and how")
    bench.add_argument("--custom", type=float, nargs="+", metavar="GB",
                       help="Additional model sizes to test (e.g., --custom 7 34 140)")

    # --- load
    load = sub.add_parser("load", help="Load a HuggingFace model with optimal strategy")
    load.add_argument("model_name", help="HuggingFace model ID (e.g., meta-llama/Llama-3-8B)")
    load.add_argument("--size", type=float, default=None, help="Model size in GB (auto-estimated if omitted)")
    load.add_argument("--chat", action="store_true", help="Start interactive chat after loading")
    load.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "plan" and args.model_size <= 0:
        parser.error("model size must be positive")

    if args.command == "plan" and hasattr(args, "moe") and args.moe:
        total_b, active_b = float(args.moe[0]), float(args.moe[1])
        n_experts, n_active = int(args.moe[2]), int(args.moe[3])
        if total_b <= 0 or active_b <= 0 or n_experts <= 0 or n_active <= 0:
            parser.error("all MoE parameters must be positive")
        if active_b > total_b:
            parser.error("active_params_b must be <= total_params_b")
        if n_active > n_experts:
            parser.error("active_experts must be <= num_experts")

    if args.command == "benchmark" and args.custom:
        for size in args.custom:
            if size <= 0:
                parser.error("custom model sizes must be positive")

    if args.command == "detect":
        hw = detect_hardware()
        print("\n=== OverflowML Hardware Detection ===")
        print(hw.summary())
        print(f"\nFor a model that needs loading, run:")
        print(f"  overflowml plan <size_in_gb>")
        print()

    elif args.command == "plan":
        hw = detect_hardware()
        print("\n=== Hardware ===")
        print(hw.summary())
        print()

        moe_profile = None
        if args.moe:
            total_b, active_b, n_experts, n_active = float(args.moe[0]), float(args.moe[1]), int(args.moe[2]), int(args.moe[3])
            # Estimate shared vs expert split (typically ~30% shared, ~70% experts for large MoE)
            shared_ratio = 0.30
            shared_gb = args.model_size * shared_ratio
            expert_gb = args.model_size * (1.0 - shared_ratio)
            moe_profile = MoEProfile(
                total_params_b=total_b,
                active_params_b=active_b,
                num_experts=n_experts,
                num_active_experts=n_active,
                shared_layers_gb=shared_gb,
                expert_size_gb=expert_gb,
            )

        strategy = pick_strategy(
            hw, args.model_size,
            prefer_speed=args.fast,
            allow_quantization=not args.no_quantize,
            moe=moe_profile,
        )
        print(f"=== Strategy for {args.model_size:.0f}GB model ===")
        if moe_profile:
            print(f"MoE: {moe_profile.total_params_b:.0f}B total, {moe_profile.active_params_b:.0f}B active, "
                  f"{moe_profile.num_experts} experts ({moe_profile.num_active_experts} active)")
            print(f"Sparsity: {moe_profile.sparsity_ratio:.0%}")
            print()
        print(strategy.summary())
        print()

        if strategy.llamacpp_flags:
            print("=== llama.cpp Launch ===")
            print("llama-server " + " ".join(strategy.llamacpp_flags) + " -m <model.gguf>")
            print()

        # Show code example
        print("=== Usage ===")
        if moe_profile:
            print("```python")
            print("import overflowml")
            print()
            print(f"moe = overflowml.MoEProfile(")
            print(f"    total_params_b={moe_profile.total_params_b}, active_params_b={moe_profile.active_params_b},")
            print(f"    num_experts={moe_profile.num_experts}, num_active_experts={moe_profile.num_active_experts},")
            print(f"    shared_layers_gb={moe_profile.shared_layers_gb:.1f}, expert_size_gb={moe_profile.expert_size_gb:.1f},")
            print(f")")
            print(f"strategy = overflowml.pick_strategy(hw, {args.model_size}, moe=moe)")
            print("```")
        else:
            print("```python")
            print("import overflowml")
            print("from diffusers import SomePipeline")
            print()
            print('pipe = SomePipeline.from_pretrained("model", torch_dtype=torch.bfloat16)')
            print(f"strategy = overflowml.optimize_pipeline(pipe, model_size_gb={args.model_size})")
            print("result = pipe(prompt)")
            print("```")
        print()

    elif args.command == "benchmark":
        _run_benchmark(args)

    elif args.command == "load":
        if args.trust_remote_code:
            print("WARNING: --trust-remote-code downloads and executes arbitrary Python "
                  "code from the model repository. Only use with models you trust.",
                  file=sys.stderr)
        from .transformers_ext import load_model
        model, tok = load_model(
            args.model_name,
            model_size_gb=args.size,
            trust_remote_code=args.trust_remote_code,
        )
        print(f"\nModel loaded: {args.model_name}")
        print(f"Type: {type(model).__name__}")
        if hasattr(model, "hf_device_map"):
            devices = set(str(v) for v in model.hf_device_map.values())
            print(f"Devices: {', '.join(sorted(devices))}")

        if args.chat:
            print("\n=== Chat (type 'quit' to exit) ===\n")
            import torch
            while True:
                try:
                    user_input = input("You: ")
                except (EOFError, KeyboardInterrupt):
                    break
                if user_input.strip().lower() in ("quit", "exit", "q"):
                    break
                inputs = tok(user_input, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs, max_new_tokens=256,
                        do_sample=True, temperature=0.7,
                    )
                response = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                print(f"AI: {response}\n")

    else:
        parser.print_help()


POPULAR_MODELS = [
    # Consumer LLMs
    ("Llama-3.2-1B", 2.5),
    ("Llama-3.2-3B", 6.5),
    ("Nemotron-Mini-4B", 8),
    ("Mistral-7B / Llama-3-8B", 16),
    ("Minitron-8B", 16),
    ("Nemotron-Nano-9B", 18),
    ("Llama-3.1-13B", 26),
    # Multi-GPU LLMs
    ("Nemotron-3 Nano 30B (MoE)", 60),
    ("Nemotron-3 Super 120B (MoE)", 120),
    ("Qwen3.5-27B (Dense)", 54),
    ("Qwen3.5-35B-A3B (MoE)", 70),
    ("Mixtral-8x7B (MoE)", 93),
    ("Step-3.5-Flash (MoE)", 392),
    ("MiniMax M2.5 (MoE)", 460),
    ("Qwen3.5-122B-A10B (MoE)", 244),
    ("Llama-3-70B", 140),
    ("NVLM-D-72B (Vision)", 156),
    ("Mixtral-8x22B (MoE)", 280),
    ("MiMo-V2-Flash (MoE)", 618),
    ("Nemotron-4-340B", 680),
    ("DeepSeek-V3.2 (MoE)", 1370),
    ("Qwen3.5-397B-A17B (MoE)", 794),
    ("GLM-5 (MoE)", 1488),
    ("Kimi K2.5 (MoE)", 2000),
    ("Llama-3.1-405B", 810),
    # Diffusers
    ("SDXL (diffusers)", 7),
    ("FLUX.1-dev (diffusers)", 24),
    ("Qwen-Image-Edit (diffusers)", 34),
    # Voice/Speech (NeMo)
    ("Parakeet-0.6B (ASR)", 1.2),
    ("Canary-1B (ASR)", 2),
    # Vision
    ("VILA-3B (Vision)", 6),
    ("VILA-8B (Vision)", 16),
    ("VILA-13B (Vision)", 26),
    ("VILA-40B (Vision)", 80),
]


def _run_benchmark(args):
    hw = detect_hardware()
    print("\n=== OverflowML Benchmark ===")
    print(hw.summary())
    print()

    models = list(POPULAR_MODELS)
    if args.custom:
        for size in args.custom:
            models.append((f"Custom ({size:.0f}GB)", size))

    name_w = max(len(m[0]) for m in models)
    header = f"{'Model':<{name_w}}  {'Size':>6}  {'Strategy':<24}  {'GPUs':>4}  {'VRAM':>8}  {'Status'}"
    print(header)
    print("-" * len(header))

    for name, size_gb in models:
        s = pick_strategy(hw, size_gb)

        # Build strategy label
        parts = []
        if s.quantization.value != "none":
            parts.append(s.quantization.value.upper())
        if s.distribution != DistributionMode.NONE:
            parts.append(f"auto ({s.num_gpus_used} GPUs)")
        if s.offload.value != "none":
            parts.append(s.offload.value.replace("_", " "))
        if s.compile:
            parts.append("compile")
        strategy_label = " + ".join(parts) if parts else "direct load"

        # Status
        if s.warnings:
            status = "!! " + s.warnings[0][:40]
        elif s.distribution != DistributionMode.NONE and s.offload != OffloadMode.NONE:
            status = "MULTI-GPU + OFFLOAD"
        elif s.distribution != DistributionMode.NONE:
            status = "MULTI-GPU"
        elif s.offload == OffloadMode.NONE:
            status = "FAST"
        elif s.offload == OffloadMode.MODEL_CPU:
            status = "OK (offload)"
        elif s.offload == OffloadMode.SEQUENTIAL_CPU:
            status = "SLOW (sequential)"
        else:
            status = "VERY SLOW (disk)"

        gpus_str = str(s.num_gpus_used)
        vram_str = f"{s.estimated_vram_gb:.1f}GB"
        print(f"{name:<{name_w}}  {size_gb:>5.0f}G  {strategy_label:<24}  {gpus_str:>4}  {vram_str:>8}  {status}")

    print()
    print("Legend: FAST = fits in VRAM | MULTI-GPU = distributed | OK = CPU offload | SLOW = layer-by-layer")
    print("Run: overflowml plan <size_gb>  for detailed strategy")
    print()


if __name__ == "__main__":
    main()
