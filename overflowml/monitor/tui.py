"""Terminal UI for live monitoring using rich."""

from __future__ import annotations

import time

from .sampler import Monitor


def run_tui(interval: float = 1.0, threshold: float = 0.85):
    """Run live terminal monitor. Requires 'rich' package."""
    try:
        from rich.live import Live
        from rich.table import Table
        from rich.console import Console
    except ImportError:
        print("Live monitor requires 'rich'. Install: pip install rich")
        return

    mon = Monitor(interval=interval, threshold=threshold)
    console = Console()

    console.print("[bold]OverflowML Monitor[/bold] — press Ctrl+C to stop\n")

    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                s = mon.sample()
                table = Table(show_header=True, header_style="bold")
                table.add_column("Metric", width=20)
                table.add_column("Value", width=30)

                if s.gpu_name:
                    table.add_row("GPU", s.gpu_name)
                    vram_pct = (s.vram_used_gb / s.vram_total_gb * 100) if s.vram_total_gb > 0 else 0
                    vram_style = "red" if vram_pct > threshold * 100 else "green"
                    table.add_row("VRAM", f"[{vram_style}]{s.vram_used_gb:.1f} / {s.vram_total_gb:.0f} GB ({vram_pct:.0f}%)[/{vram_style}]")
                else:
                    table.add_row("GPU", "None detected")

                ram_pct = (s.ram_used_gb / s.ram_total_gb * 100) if s.ram_total_gb > 0 else 0
                table.add_row("RAM", f"{s.ram_used_gb:.1f} / {s.ram_total_gb:.0f} GB ({ram_pct:.0f}%)")
                table.add_row("Samples", str(len(mon.samples)))

                warning = mon.check_threshold(s)
                if warning:
                    table.add_row("[red]WARNING[/red]", f"[red]{warning}[/red]")

                live.update(table)
                time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[dim]Monitor stopped.[/dim]")
