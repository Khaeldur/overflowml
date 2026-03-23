"""Terminal UI dashboard — interactive interface with clickable buttons.

Launch: overflowml ui
Requires: pip install overflowml[ui]  (textual)
"""

from __future__ import annotations

import sys


def run_tui():
    """Launch the interactive terminal dashboard."""
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Horizontal, Vertical, ScrollableContainer
        from textual.widgets import Button, Header, Footer, Static, Input, RichLog
        from textual.binding import Binding
    except ImportError:
        print("Terminal UI requires 'textual'. Install: pip install overflowml[ui]")
        print("  pip install textual")
        sys.exit(1)

    class OverflowMLApp(App):
        CSS = """
        Screen {
            layout: grid;
            grid-size: 2 1;
            grid-columns: 24 1fr;
        }
        #sidebar {
            dock: left;
            width: 24;
            background: $surface;
            padding: 1;
        }
        #sidebar Button {
            width: 100%;
            margin-bottom: 1;
        }
        #main {
            padding: 1;
        }
        #output {
            height: 1fr;
            border: solid green;
            padding: 1;
        }
        #input-area {
            height: 3;
            dock: bottom;
        }
        .title {
            text-align: center;
            text-style: bold;
            color: $success;
            padding: 1;
        }
        """

        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("d", "detect", "Detect"),
            Binding("o", "doctor", "Doctor"),
            Binding("p", "plan", "Plan"),
            Binding("b", "benchmark", "Benchmark"),
        ]

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal():
                with Vertical(id="sidebar"):
                    yield Static("[b]OverflowML[/b]", classes="title")
                    yield Button("Detect Hardware", id="btn-detect", variant="primary")
                    yield Button("Doctor", id="btn-doctor", variant="warning")
                    yield Button("Plan Model", id="btn-plan", variant="success")
                    yield Button("Compare Strategies", id="btn-compare", variant="success")
                    yield Button("Can Run?", id="btn-canrun")
                    yield Button("Benchmark", id="btn-benchmark")
                    yield Button("Cache", id="btn-cache")
                    yield Button("VRAM Status", id="btn-vram")
                    yield Static("")
                    yield Input(placeholder="Model ID or size (GB)...", id="model-input")
                with Vertical(id="main"):
                    yield RichLog(id="output", highlight=True, markup=True)
            yield Footer()

        def on_mount(self) -> None:
            self.title = "OverflowML"
            self.sub_title = "Runtime Control System"
            log = self.query_one("#output", RichLog)
            log.write("[bold green]OverflowML Terminal Dashboard[/bold green]")
            log.write("Click a button or use keyboard shortcuts.\n")
            log.write("Enter a model ID or size in the input field, then click Plan or Compare.\n")
            self._run_detect(log)

        def _get_model_input(self) -> str:
            inp = self.query_one("#model-input", Input)
            return inp.value.strip() or "40"

        def on_button_pressed(self, event: Button.Pressed) -> None:
            log = self.query_one("#output", RichLog)
            btn = event.button.id

            if btn == "btn-detect":
                self._run_detect(log)
            elif btn == "btn-doctor":
                self._run_doctor(log)
            elif btn == "btn-plan":
                self._run_plan(log, compare=False)
            elif btn == "btn-compare":
                self._run_plan(log, compare=True)
            elif btn == "btn-canrun":
                self._run_canrun(log)
            elif btn == "btn-benchmark":
                self._run_benchmark(log)
            elif btn == "btn-cache":
                self._run_cache(log)
            elif btn == "btn-vram":
                self._run_vram(log)

        def action_detect(self) -> None:
            self._run_detect(self.query_one("#output", RichLog))

        def action_doctor(self) -> None:
            self._run_doctor(self.query_one("#output", RichLog))

        def action_plan(self) -> None:
            self._run_plan(self.query_one("#output", RichLog), compare=False)

        def action_benchmark(self) -> None:
            self._run_benchmark(self.query_one("#output", RichLog))

        def _run_detect(self, log: RichLog):
            log.write("\n[bold cyan]== Hardware Detection ==[/bold cyan]")
            try:
                from .core.hardware import detect_hardware_info
                hw = detect_hardware_info()
                if hw.gpus:
                    for g in hw.gpus:
                        log.write(f"  GPU: {g.name} ({g.total_vram_gb:.0f}GB VRAM) [{g.backend}]")
                else:
                    log.write("  No GPU detected")
                log.write(f"  RAM: {hw.total_ram_gb:.0f}GB")
                log.write(f"  Torch: {hw.torch_version or 'not installed'}")
                if hw.torch_cuda_version:
                    log.write(f"  CUDA: {hw.torch_cuda_version}")
                log.write(f"  BF16: {'yes' if hw.supports_bf16 else 'no'} | FP8: {'yes' if hw.supports_fp8 else 'no'}")
            except Exception as e:
                log.write(f"  [red]Error: {e}[/red]")

        def _run_doctor(self, log: RichLog):
            log.write("\n[bold yellow]== Doctor ==[/bold yellow]")
            try:
                from .doctor import run as run_doctor
                report = run_doctor()

                for issue in report.issues:
                    tag = {"info": "[green]PASS[/green]", "warn": "[yellow]WARN[/yellow]", "error": "[red]FAIL[/red]"}[issue.severity]
                    log.write(f"  [{tag}] {issue.message}")
                    if issue.suggested_fix:
                        log.write(f"         Fix: {issue.suggested_fix}")

                pass_c = sum(1 for i in report.issues if i.severity == "info")
                warn_c = sum(1 for i in report.issues if i.severity == "warn")
                err_c = sum(1 for i in report.issues if i.severity == "error")
                log.write(f"\n  {pass_c} passed, {warn_c} warnings, {err_c} errors")
            except Exception as e:
                log.write(f"  [red]Error: {e}[/red]")

        def _run_plan(self, log: RichLog, compare: bool = False):
            model = self._get_model_input()
            mode = "Compare" if compare else "Plan"
            log.write(f"\n[bold green]== {mode}: {model} ==[/bold green]")
            try:
                from .core.planner import plan
                result = plan(model, compare=compare)

                if result.recommended:
                    log.write(f"  [bold]Recommended: {result.recommended.name}[/bold]")
                    log.write(f"  Speed: {result.recommended.estimated_speed}")
                    log.write(f"  Est VRAM: {result.recommended.estimated_vram_gb:.1f}GB")
                    log.write(f"  Quality: {result.recommended.quality_risk}")

                if compare and result.strategies:
                    log.write(f"\n  {'#':<3} {'Speed':<10} {'Strategy':<30} {'VRAM':>8} {'Risk':<10}")
                    log.write(f"  {'-'*65}")
                    for i, s in enumerate(result.strategies, 1):
                        if not s.viable:
                            continue
                        mark = " *" if s.recommended else ""
                        log.write(f"  {i:<3} {s.estimated_speed:<10} {s.name:<30} {s.estimated_vram_gb:>6.1f}GB {s.quality_risk:<10}{mark}")

                    rejected = [s for s in result.strategies if not s.viable]
                    if rejected:
                        log.write(f"\n  [dim]Rejected:[/dim]")
                        for s in rejected:
                            log.write(f"  [dim]  {s.name}: {s.rejection_reason}[/dim]")

                if result.explanation:
                    log.write(f"\n  [cyan]Reasoning:[/cyan]")
                    for line in result.explanation:
                        log.write(f"  {line}")
            except Exception as e:
                log.write(f"  [red]Error: {e}[/red]")

        def _run_canrun(self, log: RichLog):
            model = self._get_model_input()
            log.write(f"\n[bold]== Can Run: {model} ==[/bold]")
            try:
                from .core.can_run import can_run
                result = can_run(model)
                if result.ok:
                    log.write(f"  [green]YES[/green]: {result.reason}")
                else:
                    log.write(f"  [red]NO[/red]: {result.reason}")
                if result.recommended_strategy:
                    log.write(f"  Strategy: {result.recommended_strategy}")
                log.write(f"  VRAM: {result.detected_vram_gb:.0f}GB | RAM: {result.detected_ram_gb:.0f}GB")
            except Exception as e:
                log.write(f"  [red]Error: {e}[/red]")

        def _run_benchmark(self, log: RichLog):
            log.write("\n[bold magenta]== Benchmark ==[/bold magenta]")
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    from .detect import detect_hardware
                    from .strategy import pick_strategy, DistributionMode, OffloadMode

                hw = detect_hardware()
                models = [
                    ("Llama-3.2-1B", 2.5), ("Mistral-7B", 16), ("Llama-3.1-13B", 26),
                    ("Qwen3.5-27B", 54), ("Llama-3-70B", 140), ("FLUX.1-dev", 24),
                    ("Qwen-Image-Edit", 34), ("Llama-3.1-405B", 810),
                ]
                log.write(f"  {'Model':<20} {'Size':>6} {'Strategy':<22} {'VRAM':>8} {'Status'}")
                log.write(f"  {'-'*65}")
                for name, size in models:
                    s = pick_strategy(hw, size)
                    parts = []
                    if s.quantization.value != "none":
                        parts.append(s.quantization.value.upper())
                    if s.offload.value != "none":
                        parts.append(s.offload.value.replace("_", " "))
                    label = " + ".join(parts) if parts else "direct"
                    if s.offload == OffloadMode.NONE:
                        status = "[green]FAST[/green]"
                    elif "layer_hybrid" in s.offload.value:
                        status = "[cyan]HYBRID[/cyan]"
                    elif s.offload == OffloadMode.SEQUENTIAL_CPU:
                        status = "[yellow]SLOW[/yellow]"
                    else:
                        status = s.offload.value
                    log.write(f"  {name:<20} {size:>5.0f}G {label:<22} {s.estimated_vram_gb:>6.1f}GB {status}")
            except Exception as e:
                log.write(f"  [red]Error: {e}[/red]")

        def _run_cache(self, log: RichLog):
            log.write("\n[bold]== Cache ==[/bold]")
            try:
                from .core.cache import show_cache, CACHE_DIR
                entries = show_cache()
                if not entries:
                    log.write(f"  Cache empty ({CACHE_DIR})")
                else:
                    for e in entries:
                        status = "[green]fresh[/green]" if e.get("fresh") else "[dim]stale[/dim]"
                        log.write(f"  {e['file']}: {status}")
            except Exception as e:
                log.write(f"  [red]Error: {e}[/red]")

        def _run_vram(self, log: RichLog):
            log.write("\n[bold]== VRAM Status ==[/bold]")
            try:
                from .batch import measure_vram_headroom
                available, total = measure_vram_headroom()
                used = total - available
                pct = (used / total * 100) if total > 0 else 0
                color = "red" if pct > 85 else "yellow" if pct > 70 else "green"
                log.write(f"  [{color}]VRAM: {used:.1f}GB / {total:.0f}GB ({pct:.0f}% used)[/{color}]")
                log.write(f"  Available: {available:.1f}GB")

                from .core.runtime import diagnose_fragmentation
                frag = diagnose_fragmentation()
                if frag.is_fragmented:
                    log.write(f"  [red]Fragmentation detected: {frag.fragmentation_ratio:.0%}[/red]")
                else:
                    log.write(f"  [green]No fragmentation[/green]")
            except Exception as e:
                log.write(f"  [red]Error: {e}[/red]")

    app = OverflowMLApp()
    app.run()
