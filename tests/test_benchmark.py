"""Tests for benchmark runner — tests that don't require model downloads."""

import subprocess
import sys
from unittest.mock import MagicMock, patch

from overflowml.benchmark.runner import BenchmarkResult, _resolve_model
from overflowml.benchmark.test_models import DEFAULT_MODEL, TEXT_MODELS, BenchModel


class TestBenchModels:
    def test_default_model_exists(self):
        assert DEFAULT_MODEL is not None
        assert DEFAULT_MODEL.model_id != ""
        assert DEFAULT_MODEL.size_gb > 0

    def test_text_models_have_prompts(self):
        for m in TEXT_MODELS:
            assert m.prompt != ""
            assert m.max_new_tokens > 0

    def test_all_models_have_task(self):
        for m in TEXT_MODELS:
            assert m.task in ("text-generation", "text2text-generation")


class TestResolveModel:
    def test_none_returns_default(self):
        m = _resolve_model(None, "text-generation")
        assert m.model_id == DEFAULT_MODEL.model_id

    def test_known_model(self):
        m = _resolve_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "text-generation")
        assert m.size_gb > 0

    def test_custom_model(self):
        m = _resolve_model("my-org/my-model", "text-generation")
        assert m.model_id == "my-org/my-model"
        assert m.prompt != ""


class TestBenchmarkResult:
    def test_defaults(self):
        r = BenchmarkResult()
        assert r.tokens_per_second == 0.0
        assert r.error == ""
        assert r.prediction_matched is False

    def test_with_data(self):
        r = BenchmarkResult(
            model_id="test", tokens_generated=50,
            inference_time_s=1.0, tokens_per_second=50.0,
        )
        assert r.tokens_per_second == 50.0


class TestBenchmarkCLI:
    def test_benchmark_run_help(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "benchmark", "--help"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "--run" in r.stdout
        assert "--model" in r.stdout

    def test_benchmark_table_still_works(self):
        r = subprocess.run(
            [sys.executable, "-m", "overflowml", "benchmark"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "Legend:" in r.stdout
