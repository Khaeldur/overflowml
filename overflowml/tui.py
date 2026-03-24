"""Terminal UI dashboard — interactive interface with clickable buttons.

Launch: overflowml ui
Requires: pip install overflowml[ui]  (textual)
"""

from __future__ import annotations

import sys


def _get_app_class():
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, VerticalScroll
    from textual.widgets import Button, Header, Footer, Static, Input, Log
    from textual.binding import Binding

    class OverflowMLApp(App):
        CSS = """
        #sidebar {
            dock: left;
            width: 26;
            padding: 1;
            background: $surface;
        }
        #sidebar Button {
            width: 100%;
            margin-bottom: 1;
        }
        #model-input {
            width: 100%;
            margin-bottom: 1;
        }
        .title {
            text-align: center;
            text-style: bold;
            color: $success;
            margin-bottom: 1;
        }
        #output {
            border: solid green;
            padding: 1;
            height: 1fr;
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
            with VerticalScroll(id="sidebar"):
                yield Static("OverflowML", classes="title")
                yield Input(placeholder="Model ID or size...", id="model-input")
                yield Button("Detect Hardware", id="btn-detect", variant="primary")
                yield Button("Doctor", id="btn-doctor", variant="warning")
                yield Button("Plan Model", id="btn-plan", variant="success")
                yield Button("Compare Strategies", id="btn-compare", variant="success")
                yield Button("Can Run?", id="btn-canrun", variant="default")
                yield Button("Benchmark", id="btn-benchmark", variant="default")
                yield Button("Cache", id="btn-cache", variant="default")
                yield Button("VRAM Status", id="btn-vram", variant="default")
                yield Static("--- LLM ---", classes="title")
                yield Button("Find Servers", id="btn-discover", variant="primary")
                yield Button("Chat", id="btn-chat", variant="success")
                yield Button("Load Local Model", id="btn-load", variant="warning")
                yield Button("Unload Model", id="btn-unload", variant="error")
                yield Button("Clear Output", id="btn-clear", variant="error")
            yield Log(id="output", auto_scroll=True)
            yield Footer()

        def on_mount(self) -> None:
            self._servers = []
            self._active_server = None
            self._local_model = None
            self._chat_history = []
            self.title = "OverflowML"
            self.sub_title = "Runtime Control System"
            log = self.query_one("#output", Log)
            log.write_line("OverflowML Terminal Dashboard")
            log.write_line("Click a button or press d/o/p/b. Enter model in input field.")
            log.write_line("")
            self._run_detect(log)

        def _get_model_input(self) -> str:
            return self.query_one("#model-input", Input).value.strip() or "40"

        def on_button_pressed(self, event: Button.Pressed) -> None:
            log = self.query_one("#output", Log)
            btn = event.button.id
            if btn == "btn-clear":
                log.clear()
                log.write_line("Output cleared.")
                return

            log.write_line(f"{'=' * 50}")
            log.write_line(f"> {event.button.label}")

            try:
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
                elif btn == "btn-discover":
                    self._run_discover(log)
                elif btn == "btn-chat":
                    self._run_chat(log)
                elif btn == "btn-load":
                    self._run_load_model(log)
                elif btn == "btn-unload":
                    self._run_unload(log)
            except Exception as e:
                log.write_line(f"ERROR: {type(e).__name__}: {e}")

        def action_detect(self) -> None:
            self._run_detect(self.query_one("#output", Log))

        def action_doctor(self) -> None:
            self._run_doctor(self.query_one("#output", Log))

        def action_plan(self) -> None:
            self._run_plan(self.query_one("#output", Log), compare=False)

        def action_benchmark(self) -> None:
            self._run_benchmark(self.query_one("#output", Log))

        def _run_detect(self, log):
            log.write_line("")
            log.write_line("== Hardware Detection ==")
            try:
                from overflowml.core.hardware import detect_hardware_info
                hw = detect_hardware_info()
                if hw.gpus:
                    for g in hw.gpus:
                        log.write_line(f"  GPU: {g.name} ({g.total_vram_gb:.0f}GB VRAM) [{g.backend}]")
                else:
                    log.write_line("  No GPU detected")
                log.write_line(f"  RAM: {hw.total_ram_gb:.0f}GB")
                log.write_line(f"  Torch: {hw.torch_version or 'not installed'}")
                if hw.torch_cuda_version:
                    log.write_line(f"  CUDA: {hw.torch_cuda_version}")
                log.write_line(f"  BF16: {'yes' if hw.supports_bf16 else 'no'} | FP8: {'yes' if hw.supports_fp8 else 'no'}")
            except Exception as e:
                log.write_line(f"  Error: {e}")

        def _run_doctor(self, log):
            log.write_line("")
            log.write_line("== Doctor ==")
            try:
                from overflowml.doctor import run as run_doctor
                report = run_doctor()

                for issue in report.issues:
                    tag = {"info": "PASS", "warn": "WARN", "error": "FAIL"}[issue.severity]
                    log.write_line(f"  [{tag}] {issue.message}")
                    if issue.suggested_fix:
                        log.write_line(f"         Fix: {issue.suggested_fix}")

                pass_c = sum(1 for i in report.issues if i.severity == "info")
                warn_c = sum(1 for i in report.issues if i.severity == "warn")
                err_c = sum(1 for i in report.issues if i.severity == "error")
                log.write_line(f"  {pass_c} passed, {warn_c} warnings, {err_c} errors")
            except Exception as e:
                log.write_line(f"  Error: {e}")

        def _run_plan(self, log, compare=False):
            model = self._get_model_input()
            mode = "Compare" if compare else "Plan"
            log.write_line("")
            log.write_line(f"== {mode}: {model} ==")
            try:
                from overflowml.core.planner import plan
                result = plan(model, compare=compare)

                if result.recommended:
                    log.write_line(f"  Recommended: {result.recommended.name}")
                    log.write_line(f"  Speed: {result.recommended.estimated_speed}")
                    log.write_line(f"  Est VRAM: {result.recommended.estimated_vram_gb:.1f}GB")
                    log.write_line(f"  Quality: {result.recommended.quality_risk}")

                if compare and result.strategies:
                    log.write_line("")
                    log.write_line(f"  {'#':<3} {'Speed':<10} {'Strategy':<30} {'VRAM':>8} {'Risk':<10}")
                    log.write_line(f"  {'-'*65}")
                    for i, s in enumerate(result.strategies, 1):
                        if not s.viable:
                            continue
                        mark = " *" if s.recommended else ""
                        log.write_line(f"  {i:<3} {s.estimated_speed:<10} {s.name:<30} {s.estimated_vram_gb:>6.1f}GB {s.quality_risk:<10}{mark}")

                    rejected = [s for s in result.strategies if not s.viable]
                    if rejected:
                        log.write_line(f"  Rejected:")
                        for s in rejected:
                            log.write_line(f"    {s.name}: {s.rejection_reason}")

                if result.explanation:
                    log.write_line("")
                    log.write_line("  Reasoning:")
                    for line in result.explanation:
                        log.write_line(f"  {line}")
            except Exception as e:
                log.write_line(f"  Error: {e}")

        def _run_canrun(self, log):
            model = self._get_model_input()
            log.write_line("")
            log.write_line(f"== Can Run: {model} ==")
            try:
                from overflowml.core.can_run import can_run
                result = can_run(model)
                status = "YES" if result.ok else "NO"
                log.write_line(f"  {status}: {result.reason}")
                if result.recommended_strategy:
                    log.write_line(f"  Strategy: {result.recommended_strategy}")
                log.write_line(f"  VRAM: {result.detected_vram_gb:.0f}GB | RAM: {result.detected_ram_gb:.0f}GB")
            except Exception as e:
                log.write_line(f"  Error: {e}")

        def _run_benchmark(self, log):
            log.write_line("")
            log.write_line("== Benchmark ==")
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    from overflowml.detect import detect_hardware
                    from overflowml.strategy import pick_strategy, OffloadMode

                hw = detect_hardware()
                models = [
                    ("Llama-3.2-1B", 2.5), ("Mistral-7B", 16), ("Llama-3.1-13B", 26),
                    ("Qwen3.5-27B", 54), ("Llama-3-70B", 140), ("FLUX.1-dev", 24),
                    ("Qwen-Image-Edit", 34), ("Llama-3.1-405B", 810),
                ]
                log.write_line(f"  {'Model':<20} {'Size':>6} {'Strategy':<22} {'VRAM':>8} {'Status'}")
                log.write_line(f"  {'-'*70}")
                for name, size in models:
                    s = pick_strategy(hw, size)
                    parts = []
                    if s.quantization.value != "none":
                        parts.append(s.quantization.value.upper())
                    if s.offload.value != "none":
                        parts.append(s.offload.value.replace("_", " "))
                    label = " + ".join(parts) if parts else "direct"
                    if s.offload == OffloadMode.NONE:
                        status = "FAST"
                    elif "layer_hybrid" in s.offload.value:
                        status = "HYBRID"
                    elif s.offload == OffloadMode.SEQUENTIAL_CPU:
                        status = "SLOW"
                    else:
                        status = s.offload.value
                    log.write_line(f"  {name:<20} {size:>5.0f}G {label:<22} {s.estimated_vram_gb:>6.1f}GB {status}")
            except Exception as e:
                log.write_line(f"  Error: {e}")

        def _run_cache(self, log):
            log.write_line("")
            log.write_line("== Cache ==")
            try:
                from overflowml.core.cache import show_cache, CACHE_DIR
                entries = show_cache()
                if not entries:
                    log.write_line(f"  Cache empty ({CACHE_DIR})")
                else:
                    for e in entries:
                        status = "fresh" if e.get("fresh") else "stale"
                        log.write_line(f"  {e['file']}: {status}")
            except Exception as e:
                log.write_line(f"  Error: {e}")

        def _run_vram(self, log):
            log.write_line("")
            log.write_line("== VRAM Status ==")
            try:
                from overflowml.batch import measure_vram_headroom
                available, total = measure_vram_headroom()
                used = total - available
                pct = (used / total * 100) if total > 0 else 0
                log.write_line(f"  VRAM: {used:.1f}GB / {total:.0f}GB ({pct:.0f}% used)")
                log.write_line(f"  Available: {available:.1f}GB")

                from overflowml.core.runtime import diagnose_fragmentation
                frag = diagnose_fragmentation()
                if frag.is_fragmented:
                    log.write_line(f"  WARNING: Fragmentation {frag.fragmentation_ratio:.0%}")
                else:
                    log.write_line(f"  No fragmentation")
            except Exception as e:
                log.write_line(f"  Error: {e}")

        def _run_discover(self, log):
            log.write_line("")
            log.write_line("== Scanning for LLM Servers ==")
            try:
                from overflowml.inference import discover_servers, list_models
                self._servers = discover_servers(timeout=2.0)
                if not self._servers:
                    log.write_line("  No servers found on common ports.")
                    log.write_line("  Start Ollama, llama.cpp, or vLLM first.")
                    return

                for i, s in enumerate(self._servers):
                    models = getattr(s, "_models", []) or list_models(s)
                    model_str = ", ".join(models[:5]) if models else "none"
                    log.write_line(f"  [{i+1}] {s.name} ({s.url}) - {s.backend}")
                    log.write_line(f"      Models: {model_str}")

                # Auto-select first server
                self._active_server = self._servers[0]
                if not self._active_server.model:
                    models = getattr(self._active_server, "_models", [])
                    if models:
                        self._active_server.model = models[0]
                log.write_line(f"  Active: {self._active_server.name} / {self._active_server.model}")
                log.write_line("  Type a message in the input field and click Chat.")
            except Exception as e:
                log.write_line(f"  Error: {e}")

        def _run_chat(self, log):
            user_msg = self._get_model_input()
            if not user_msg or user_msg == "40":
                log.write_line("")
                log.write_line("  Type a message in the input field first, then click Chat.")
                return

            log.write_line("")
            log.write_line(f"  You: {user_msg}")

            # Try server first, then local model
            if self._active_server:
                try:
                    from overflowml.inference import chat, ChatMessage
                    self._chat_history.append(ChatMessage(role="user", content=user_msg))
                    response = chat(self._active_server, self._chat_history)
                    if response.error:
                        log.write_line(f"  Error: {response.error}")
                    else:
                        log.write_line(f"  AI ({response.model}): {response.content}")
                        self._chat_history.append(ChatMessage(role="assistant", content=response.content))
                        if response.tokens_used:
                            log.write_line(f"  [{response.tokens_used} tokens]")
                except Exception as e:
                    log.write_line(f"  Error: {e}")
            elif self._local_model and self._local_model.loaded:
                response = self._local_model.chat(user_msg)
                if response.error:
                    log.write_line(f"  Error: {response.error}")
                else:
                    log.write_line(f"  AI ({response.model}): {response.content}")
            else:
                log.write_line("  No server connected and no model loaded.")
                log.write_line("  Click 'Find Servers' or 'Load Local Model' first.")

            # Clear input
            self.query_one("#model-input", Input).value = ""

        def _run_load_model(self, log):
            model_id = self._get_model_input()
            log.write_line("")
            log.write_line(f"== Loading Model: {model_id} ==")
            log.write_line("  This may take a while (downloading + loading)...")
            try:
                from overflowml.inference import LocalModel
                self._local_model = LocalModel()
                status = self._local_model.load(model_id)
                log.write_line(f"  {status}")
                if self._local_model.loaded:
                    log.write_line("  Type a message and click Chat.")
            except Exception as e:
                log.write_line(f"  Error: {e}")

        def _run_unload(self, log):
            log.write_line("")
            if self._local_model and self._local_model.loaded:
                name = self._local_model.model_name
                self._local_model.unload()
                log.write_line(f"  Unloaded {name}. VRAM freed.")
            else:
                log.write_line("  No model loaded.")
            self._active_server = None
            self._chat_history = []
            log.write_line("  Chat history cleared.")

    return OverflowMLApp


def run_tui():
    """Launch the interactive terminal dashboard."""
    try:
        AppClass = _get_app_class()
    except ImportError:
        print("Terminal UI requires 'textual'. Install: pip install overflowml[ui]")
        print("  pip install textual")
        sys.exit(1)
    app = AppClass()
    app.run()
