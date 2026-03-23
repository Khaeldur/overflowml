"""Tests for monitor sampler and metrics exporter."""

from overflowml.monitor.sampler import Monitor, Sample


class TestMonitor:
    def test_sample(self):
        mon = Monitor()
        s = mon.sample()
        assert isinstance(s, Sample)
        assert s.timestamp > 0
        assert s.ram_total_gb > 0

    def test_samples_accumulate(self):
        mon = Monitor()
        mon.sample()
        mon.sample()
        assert len(mon.samples) == 2

    def test_threshold_check_no_gpu(self):
        mon = Monitor(threshold=0.85)
        s = mon.sample()
        if s.vram_total_gb == 0:
            assert mon.check_threshold(s) is None

    def test_threshold_configurable(self):
        mon = Monitor(threshold=0.5)
        assert mon.threshold == 0.5


class TestMetricsExporter:
    def test_import(self):
        from overflowml.monitor.metrics import MetricsExporter
        e = MetricsExporter(port=19108)
        assert e.port == 19108
        assert e.interval == 5.0
