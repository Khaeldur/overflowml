"""Prometheus metrics exporter for production observability."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from .sampler import Monitor

logger = logging.getLogger("overflowml")


class MetricsExporter:
    """Export VRAM/RAM metrics as Prometheus gauges.

    Usage:
        from overflowml.monitor import MetricsExporter
        exporter = MetricsExporter(port=9108)
        exporter.start()
    """

    def __init__(self, port: int = 9108, interval: float = 5.0):
        self.port = port
        self.interval = interval
        self._monitor = Monitor(interval=interval)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._gauges_initialized = False

    def start(self):
        """Start the metrics server and background sampler."""
        try:
            from prometheus_client import start_http_server, Gauge
        except ImportError:
            logger.error("prometheus_client not installed. Install: pip install prometheus-client")
            return

        self._vram_used = Gauge("overflowml_vram_used_bytes", "GPU VRAM used in bytes")
        self._vram_total = Gauge("overflowml_vram_total_bytes", "GPU VRAM total in bytes")
        self._ram_used = Gauge("overflowml_ram_used_bytes", "System RAM used in bytes")
        self._ram_total = Gauge("overflowml_ram_total_bytes", "System RAM total in bytes")
        self._near_oom = Gauge("overflowml_near_oom", "1 if VRAM above threshold, 0 otherwise")
        self._gauges_initialized = True

        start_http_server(self.port)
        logger.info("Prometheus metrics available at http://localhost:%d/metrics", self.port)

        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background sampler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _sample_loop(self):
        while self._running:
            s = self._monitor.sample()
            if self._gauges_initialized:
                self._vram_used.set(s.vram_used_gb * (1024 ** 3))
                self._vram_total.set(s.vram_total_gb * (1024 ** 3))
                self._ram_used.set(s.ram_used_gb * (1024 ** 3))
                self._ram_total.set(s.ram_total_gb * (1024 ** 3))
                warning = self._monitor.check_threshold(s)
                self._near_oom.set(1 if warning else 0)
            time.sleep(self.interval)
