"""CLI tool for hardware detection, model inspection, and strategy planning."""

import argparse
import json
import logging
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import warnings as _w
with _w.catch_warnings():
    _w.simplefilter("ignore", DeprecationWarning)
    from .detect import detect_hardware
    from .strategy import DistributionMode, MoEProfile, OffloadMode, pick_strategy, plan_llamacpp


def main():
    parser = argparse.ArgumentParser(
        prog="overflowml",
        description="OverflowML — Run AI models larger than your GPU",
        epilog="Examples:\n"
               "  overflowml detect\n"
               "  overflowml inspect meta-llama/Llama-3-70B\n"
               "  overflowml plan 40\n"
               "  overflowml plan meta-llama/Llama-3-70B --compare\n"
               "  overflowml doctor\n"
               "  overflowml benchmark --custom 70 140\n"
               "  overflowml load meta-llama/Llama-3-8B --chat\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # --- detect
    sub.add_parser("detect", help="Detect hardware and show capabilities")

    # --- inspect
    insp = sub.add_parser("inspect", help="Inspect a model and estimate memory footprint")
    insp.add_argument("model_id", help="HuggingFace model ID (e.g., meta-llama/Llama-3-70B)")
    insp.add_argument("--trust-remote-code", action="store_true")
    insp.add_argument("--json", dest="json_output", action="store_true", help="Output as JSON")

    # --- plan
    plan = sub.add_parser("plan", help="Plan optimal loading strategy for a model")
    plan.add_argument("model_size", type=str, help="Model size in GB or HuggingFace model ID")
    plan.add_argument("--fast", action="store_true", help="Prefer speed over VRAM savings")
    plan.add_argument("--no-quantize", action="store_true", help="Disable quantization")
    plan.add_argument("--compare", action="store_true", help="Show all viable strategies")
    plan.add_argument("--assume-size-gb", type=float, default=None, help="Override auto-detected model size")
    plan.add_argument("--lora-size-gb", type=float, default=None, help="LoRA adapter size in GB (added to VRAM estimate)")
    plan.add_argument("--json", dest="json_output", action="store_true", help="Output as JSON")
    plan.add_argument("--moe", nargs=4, metavar=("TOTAL_B", "ACTIVE_B", "EXPERTS", "ACTIVE_EXPERTS"),
                       help="MoE config: total_params_B active_params_B num_experts active_experts")
    plan.add_argument("--trust-remote-code", action="store_true")

    # --- doctor
    doc = sub.add_parser("doctor", help="Check environment health for AI model loading")
    doc.add_argument("--model", type=str, default=None, help="Check fit for a specific model ID")
    doc.add_argument("--model-size-gb", type=float, default=None, help="Check fit for a specific size")
    doc.add_argument("--json", dest="json_output", action="store_true", help="Output as JSON")

    # --- can-run
    canrun = sub.add_parser("can-run", help="Check if a model can run on this hardware (CI/CD gating)")
    canrun.add_argument("model_size", type=str, help="Model size in GB or HuggingFace model ID")
    canrun.add_argument("--max-offload", type=str, default="sequential_cpu",
                        choices=["none", "model_cpu", "sequential_cpu", "disk"],
                        help="Maximum acceptable offload mode (default: sequential_cpu)")
    canrun.add_argument("--json", dest="json_output", action="store_true", help="Output as JSON")
    canrun.add_argument("--trust-remote-code", action="store_true")

    # --- cache
    cache_parser = sub.add_parser("cache", help="Manage strategy cache")
    cache_sub = cache_parser.add_subparsers(dest="cache_action")
    cache_sub.add_parser("show", help="Show cached entries")
    cache_sub.add_parser("clear", help="Clear all cached data")

    # --- monitor
    mon = sub.add_parser("monitor", help="Live VRAM/RAM monitoring dashboard")
    mon.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds")
    mon.add_argument("--threshold", type=float, default=0.85, help="VRAM warning threshold (0-1)")

    # --- benchmark
    bench = sub.add_parser("benchmark", help="Show what models your hardware can run and how")
    bench.add_argument("--custom", type=float, nargs="+", metavar="GB",
                       help="Additional model sizes to test (e.g., --custom 7 34 140)")
    bench.add_argument("--run", action="store_true",
                       help="Run real inference benchmark (downloads a small test model)")
    bench.add_argument("--model", type=str, default=None,
                       help="HuggingFace model ID for --run (default: TinyLlama-1.1B)")
    bench.add_argument("--json", dest="json_output", action="store_true",
                       help="Output as JSON (with --run)")
    bench.add_argument("--trust-remote-code", action="store_true")

    # --- load
    load = sub.add_parser("load", help="Load a HuggingFace model with optimal strategy")
    load.add_argument("model_name", help="HuggingFace model ID (e.g., meta-llama/Llama-3-8B)")
    load.add_argument("--size", type=float, default=None, help="Model size in GB (auto-estimated if omitted)")
    load.add_argument("--chat", action="store_true", help="Start interactive chat after loading")
    load.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # --- Validation ---
    if args.command == "plan":
        _validate_plan_args(args, parser)
    if args.command == "benchmark" and hasattr(args, "custom") and args.custom:
        for size in args.custom:
            if size <= 0:
                parser.error("custom model sizes must be positive")

    # --- Dispatch ---
    if args.command == "detect":
        _cmd_detect()
    elif args.command == "inspect":
        _cmd_inspect(args)
    elif args.command == "plan":
        _cmd_plan(args)
    elif args.command == "doctor":
        _cmd_doctor(args)
    elif args.command == "can-run":
        _cmd_can_run(args)
    elif args.command == "monitor":
        from .monitor.tui import run_tui
        run_tui(interval=args.interval, threshold=args.threshold)
    elif args.command == "cache":
        _cmd_cache(args)
    elif args.command == "benchmark":
        if getattr(args, "run", False):
            _cmd_benchmark_run(args)
        else:
            _run_benchmark(args)
    elif args.command == "load":
        _cmd_load(args)
    else:
        parser.print_help()


def _validate_plan_args(args, parser):
    # Check if model_size is numeric
    try:
        size = float(args.model_size)
        if size <= 0:
            parser.error("model size must be positive")
    except ValueError:
        pass  # it's a model ID, validated later

    if args.moe:
        # MoE requires numeric model_size
        try:
            float(args.model_size)
        except ValueError:
            parser.error("--moe requires a numeric model size (e.g., overflowml plan 120 --moe ...)")
        total_b, active_b = float(args.moe[0]), float(args.moe[1])
        n_experts, n_active = int(args.moe[2]), int(args.moe[3])
        if total_b <= 0 or active_b <= 0 or n_experts <= 0 or n_active <= 0:
            parser.error("all MoE parameters must be positive")
        if active_b > total_b:
            parser.error("active_params_b must be <= total_params_b")
        if n_active > n_experts:
            parser.error("active_experts must be <= num_experts")


# ---- Command handlers ----

def _cmd_detect():
    hw = detect_hardware()
    print("\n=== OverflowML Hardware Detection ===")
    print(hw.summary())
    print(f"\nFor a model that needs loading, run:")
    print(f"  overflowml plan <model_or_size>")
    print()


def _cmd_inspect(args):
    from .inspect import inspect_model
    info = inspect_model(args.model_id, trust_remote_code=args.trust_remote_code)

    if args.json_output:
        import dataclasses
        print(json.dumps(dataclasses.asdict(info), indent=2))
        return

    print(f"\nModel: {info.model_id}")
    if info.architecture:
        print(f"Architecture: {info.architecture}")
    if info.task_family != "unknown":
        print(f"Task: {info.task_family}")
    if info.param_count:
        print(f"Estimated params: {info.param_count / 1e9:.1f}B")
    if info.estimated_sizes_gb:
        print("Estimated weights:")
        for dtype, size in info.estimated_sizes_gb.items():
            print(f"  {dtype:>5}: {size:>7.1f} GB")
    print(f"Source: {info.source}")
    print(f"Confidence: {info.confidence}")
    if info.notes:
        for n in info.notes:
            print(f"  {n}")
    print()


def _cmd_plan(args):
    from .core.planner import plan as do_plan

    # Resolve model_size: numeric or model ID
    model_or_size = args.model_size
    try:
        model_or_size = float(args.model_size)
    except ValueError:
        pass  # it's a model ID string

    if args.assume_size_gb:
        model_or_size = args.assume_size_gb

    # MoE path: use legacy planner directly
    if args.moe:
        _cmd_plan_legacy_moe(args)
        return

    result = do_plan(
        model_or_size,
        compare=args.compare,
        trust_remote_code=getattr(args, "trust_remote_code", False),
        lora_size_gb=getattr(args, "lora_size_gb", None),
    )

    if args.json_output:
        import dataclasses
        out = dataclasses.asdict(result)
        print(json.dumps(out, indent=2, default=str))
        return

    # Detect model size for display
    if result.model and result.model.estimated_sizes_gb:
        fp16 = result.model.estimated_sizes_gb.get("fp16", 0)
        if result.model.model_id and not result.model.model_id.replace(".", "").replace("-", "").isdigit():
            print(f"\nDetected: {result.model.model_id} (~{fp16:.0f}GB fp16)")

    # Hardware
    if result.hardware and result.hardware.gpus:
        gpu = result.hardware.gpus[0]
        print(f"Hardware: {gpu.name} ({gpu.total_vram_gb:.0f}GB VRAM), {result.hardware.total_ram_gb:.0f}GB RAM")

    if args.compare and result.strategies:
        _print_compare_table(result)
    elif result.recommended:
        # Single strategy output
        size_str = f"{model_or_size}" if isinstance(model_or_size, (int, float)) else result.model.model_id if result.model else str(model_or_size)
        print(f"\n=== Recommended Strategy ===")
        print(f"  {result.recommended.name}")
        print(f"  Speed: {result.recommended.estimated_speed}")
        print(f"  Est VRAM: {result.recommended.estimated_vram_gb:.1f}GB")
        print(f"  Quality risk: {result.recommended.quality_risk}")
    else:
        print("\nNo viable strategy found.")

    # Reasoning
    if result.explanation:
        print(f"\n=== Reasoning ===")
        for line in result.explanation:
            print(f"  {line}")
    print()


def _print_compare_table(result):
    print(f"\n=== Viable Strategies ===")
    # Header
    print(f"{'#':<3} {'Speed':<10} {'Strategy':<35} {'Est VRAM':>10} {'Quality Risk':<15}")
    print("-" * 75)
    for i, s in enumerate(result.strategies, 1):
        if not s.viable:
            continue
        marker = " <- recommended" if s.recommended else ""
        print(f"{i:<3} {s.estimated_speed:<10} {s.name:<35} {s.estimated_vram_gb:>8.1f}GB {s.quality_risk:<15}{marker}")

    # Show rejected
    rejected = [s for s in result.strategies if not s.viable]
    if rejected:
        print(f"\nRejected:")
        for s in rejected:
            print(f"  {s.name}: {s.rejection_reason}")


def _cmd_plan_legacy_moe(args):
    """Handle MoE planning using legacy strategy path."""
    hw = detect_hardware()
    model_size = float(args.model_size)

    print("\n=== Hardware ===")
    print(hw.summary())
    print()

    total_b, active_b, n_experts, n_active = float(args.moe[0]), float(args.moe[1]), int(args.moe[2]), int(args.moe[3])
    shared_ratio = 0.30
    moe_profile = MoEProfile(
        total_params_b=total_b, active_params_b=active_b,
        num_experts=n_experts, num_active_experts=n_active,
        shared_layers_gb=model_size * shared_ratio,
        expert_size_gb=model_size * (1.0 - shared_ratio),
    )

    strategy = pick_strategy(hw, model_size, prefer_speed=args.fast,
                              allow_quantization=not args.no_quantize, moe=moe_profile)

    print(f"=== Strategy for {model_size:.0f}GB model ===")
    print(f"MoE: {total_b:.0f}B total, {active_b:.0f}B active, {n_experts} experts ({n_active} active)")
    print(f"Sparsity: {moe_profile.sparsity_ratio:.0%}")
    print()
    print(strategy.summary(include_notes=False))

    if strategy.notes:
        print(f"\n=== Reasoning ===")
        for n in strategy.notes:
            print(f"  - {n}")

    if strategy.llamacpp_flags:
        print("\n=== llama.cpp Launch ===")
        print("llama-server " + " ".join(strategy.llamacpp_flags) + " -m <model.gguf>")
    print()


def _cmd_doctor(args):
    from .doctor import run as run_doctor

    report = run_doctor(
        model=args.model,
        model_size_gb=args.model_size_gb,
    )

    if args.json_output:
        import dataclasses
        out = dataclasses.asdict(report)
        print(json.dumps(out, indent=2, default=str))
        return

    print("\n=== OverflowML Doctor ===\n")

    # Environment
    print("Environment")
    for k, v in report.environment.items():
        print(f"  {k}: {v}")
    print()

    # Hardware
    print("Hardware")
    for k, v in report.hardware.items():
        print(f"  {k}: {v}")
    print()

    # Issues / checks
    if report.issues:
        print("Checks")
        pass_count = sum(1 for i in report.issues if i.severity == "info")
        warn_count = sum(1 for i in report.issues if i.severity == "warn")
        err_count = sum(1 for i in report.issues if i.severity == "error")

        for issue in report.issues:
            tag = {"info": "PASS", "warn": "WARN", "error": "FAIL"}[issue.severity]
            print(f"  [{tag}] {issue.message}")
            if issue.suggested_fix:
                print(f"         Fix: {issue.suggested_fix}")

        print(f"\n{pass_count} passed, {warn_count} warnings, {err_count} errors")
    else:
        print("No issues detected.")

    # Fix commands
    if report.fix_commands:
        print(f"\nSuggested fixes:")
        for cmd in report.fix_commands:
            print(f"  {cmd}")
    print()


def _cmd_benchmark_run(args):
    from .benchmark import run_benchmark

    print("\n=== OverflowML Real Inference Benchmark ===\n")
    result = run_benchmark(
        model_id=args.model,
        trust_remote_code=getattr(args, "trust_remote_code", False),
    )

    if getattr(args, "json_output", False):
        import dataclasses
        print(json.dumps(dataclasses.asdict(result), indent=2, default=str))
        return

    if result.error:
        print(f"ERROR: {result.error}")
        return

    print(f"Model: {result.model_id}")
    print(f"Strategy: {result.strategy_used}")
    print(f"Load time: {result.load_time_s:.1f}s")
    print(f"Warmup: {result.warmup_time_s:.2f}s")
    print(f"Inference: {result.inference_time_s:.2f}s")
    print(f"Tokens generated: {result.tokens_generated}")
    print(f"Throughput: {result.tokens_per_second:.1f} tok/s")
    if result.peak_vram_gb > 0:
        print(f"Peak VRAM: {result.peak_vram_gb:.2f} GB")
    if result.predicted_strategy:
        match = "YES" if result.prediction_matched else "NO"
        print(f"\nPredicted strategy: {result.predicted_strategy}")
        print(f"Prediction matched: {match}")
    for note in result.notes:
        print(f"  {note}")
    print()


def _cmd_cache(args):
    from .core.cache import clear_cache, show_cache, CACHE_DIR

    if args.cache_action == "clear":
        count = clear_cache()
        print(f"Cleared {count} cached entries from {CACHE_DIR}")
    elif args.cache_action == "show":
        entries = show_cache()
        if not entries:
            print(f"Cache empty ({CACHE_DIR})")
            return
        print(f"Cache: {CACHE_DIR}\n")
        for e in entries:
            if "error" in e:
                print(f"  {e['file']}: {e['error']}")
            else:
                status = "fresh" if e["fresh"] else "stale"
                print(f"  {e['file']}: v{e['version']}, {e['age_seconds']}s old ({status})")
    else:
        print("Usage: overflowml cache show | overflowml cache clear")


def _cmd_can_run(args):
    from .core.can_run import can_run

    model_or_size = args.model_size
    try:
        model_or_size = float(args.model_size)
    except ValueError:
        pass

    result = can_run(
        model_or_size,
        max_offload=args.max_offload,
        trust_remote_code=getattr(args, "trust_remote_code", False),
    )

    if args.json_output:
        import dataclasses
        print(json.dumps(dataclasses.asdict(result), indent=2, default=str))
        return

    status = "YES" if result.ok else "NO"
    print(f"\n{status}: {result.reason}")
    if result.recommended_strategy:
        print(f"Strategy: {result.recommended_strategy}")
    print(f"Hardware: {result.detected_vram_gb:.0f}GB VRAM, {result.detected_ram_gb:.0f}GB RAM")

    if not result.ok:
        sys.exit(1)
    print()


def _cmd_load(args):
    if args.trust_remote_code:
        print("WARNING: --trust-remote-code downloads and executes arbitrary Python "
              "code from the model repository. Only use with models you trust.",
              file=sys.stderr)
    from .transformers_ext import load_model
    model, tok = load_model(
        args.model_name, model_size_gb=args.size,
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
                outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
            response = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"AI: {response}\n")


# ---- Benchmark (legacy) ----

POPULAR_MODELS = [
    ("Llama-3.2-1B", 2.5),
    ("Llama-3.2-3B", 6.5),
    ("Nemotron-Mini-4B", 8),
    ("Mistral-7B / Llama-3-8B", 16),
    ("Minitron-8B", 16),
    ("Nemotron-Nano-9B", 18),
    ("Llama-3.1-13B", 26),
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
    ("SDXL (diffusers)", 7),
    ("FLUX.1-dev (diffusers)", 24),
    ("Qwen-Image-Edit (diffusers)", 34),
    ("Parakeet-0.6B (ASR)", 1.2),
    ("Canary-1B (ASR)", 2),
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
