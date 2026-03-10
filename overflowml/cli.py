"""CLI tool for hardware detection and strategy recommendation."""

import argparse
import logging
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from .detect import detect_hardware
from .strategy import pick_strategy


def main():
    parser = argparse.ArgumentParser(
        prog="overflowml",
        description="OverflowML — Run AI models larger than your GPU",
    )
    sub = parser.add_subparsers(dest="command")

    # --- detect
    sub.add_parser("detect", help="Detect hardware and show capabilities")

    # --- plan
    plan = sub.add_parser("plan", help="Plan optimal loading strategy for a model")
    plan.add_argument("model_size", type=float, help="Model size in GB (BF16 weights)")
    plan.add_argument("--fast", action="store_true", help="Prefer speed over VRAM savings")
    plan.add_argument("--no-quantize", action="store_true", help="Disable quantization")

    # --- load
    load = sub.add_parser("load", help="Load a HuggingFace model with optimal strategy")
    load.add_argument("model_name", help="HuggingFace model ID (e.g., meta-llama/Llama-3-8B)")
    load.add_argument("--size", type=float, default=None, help="Model size in GB (auto-estimated if omitted)")
    load.add_argument("--chat", action="store_true", help="Start interactive chat after loading")
    load.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

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

        strategy = pick_strategy(
            hw, args.model_size,
            prefer_speed=args.fast,
            allow_quantization=not args.no_quantize,
        )
        print(f"=== Strategy for {args.model_size:.0f}GB model ===")
        print(strategy.summary())
        print()

        # Show code example
        print("=== Usage ===")
        print("```python")
        print("import overflowml")
        print("from diffusers import SomePipeline")
        print()
        print('pipe = SomePipeline.from_pretrained("model", torch_dtype=torch.bfloat16)')
        print(f"strategy = overflowml.optimize_pipeline(pipe, model_size_gb={args.model_size})")
        print("result = pipe(prompt)")
        print("```")
        print()

    elif args.command == "load":
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


if __name__ == "__main__":
    main()
