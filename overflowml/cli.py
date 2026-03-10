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

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
