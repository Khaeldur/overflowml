"""Tests for model inspection and size estimation."""

from unittest.mock import MagicMock, patch

from overflowml.inspect.arch_registry import classify_task, estimate_params_from_config
from overflowml.inspect.model_estimator import inspect_model, estimate_size_gb
from overflowml.core.types import ModelInfo


class TestClassifyTask:
    def test_causal_lm(self):
        assert classify_task("LlamaForCausalLM") == "causal-lm"

    def test_diffusers(self):
        assert classify_task("UNet2DConditionModel", "stable-diffusion-xl") == "diffusers"

    def test_embedding(self):
        assert classify_task("", "bge-large-en-v1.5") == "embedding"

    def test_unknown(self):
        assert classify_task("SomeRandomArch") == "unknown"

    def test_mistral(self):
        assert classify_task("MistralForCausalLM") == "causal-lm"


class TestEstimateParamsFromConfig:
    def test_explicit_param_count(self):
        config = {"num_parameters": 70_000_000_000}
        count, source = estimate_params_from_config(config)
        assert count == 70_000_000_000
        assert "explicit" in source

    def test_architecture_estimate(self):
        config = {"hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 32000}
        count, source = estimate_params_from_config(config)
        assert count > 0
        assert "architecture" in source

    def test_moe_multiplier(self):
        config = {
            "hidden_size": 4096, "num_hidden_layers": 32,
            "vocab_size": 32000, "num_local_experts": 8,
        }
        count_moe, _ = estimate_params_from_config(config)
        config_dense = {
            "hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 32000,
        }
        count_dense, _ = estimate_params_from_config(config_dense)
        assert count_moe > count_dense

    def test_missing_fields(self):
        count, source = estimate_params_from_config({})
        assert count is None
        assert source == "unknown"


class TestInspectModel:
    @patch("overflowml.inspect.model_estimator.probe_safetensors_size", return_value=140_000_000_000)
    @patch("overflowml.inspect.model_estimator.probe_config", return_value={
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 8192,
    })
    def test_safetensors_path(self, mock_config, mock_st):
        info = inspect_model("test/model")
        assert info.confidence == "high"
        assert info.source == "safetensors index"
        assert "fp16" in info.estimated_sizes_gb
        assert info.architecture == "LlamaForCausalLM"

    @patch("overflowml.inspect.model_estimator.probe_safetensors_size", return_value=None)
    @patch("overflowml.inspect.model_estimator.probe_config", return_value={
        "architectures": ["MistralForCausalLM"],
        "hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 32000,
    })
    def test_config_fallback(self, mock_config, mock_st):
        info = inspect_model("test/model")
        assert info.confidence == "medium"
        assert "config.json" in info.source
        assert info.param_count > 0

    @patch("overflowml.inspect.model_estimator.probe_safetensors_size", return_value=None)
    @patch("overflowml.inspect.model_estimator.probe_config", return_value=None)
    def test_no_data(self, mock_config, mock_st):
        info = inspect_model("test/nonexistent")
        assert info.confidence == "low"
        assert info.param_count is None

    @patch("overflowml.inspect.model_estimator.probe_safetensors_size", return_value=16_000_000_000)
    @patch("overflowml.inspect.model_estimator.probe_config", return_value=None)
    def test_size_math(self, mock_config, mock_st):
        info = inspect_model("test/model")
        # 16B bytes / 2 = 8B params
        assert info.param_count == 8_000_000_000
        assert info.estimated_sizes_gb["fp16"] > 14.0


class TestEstimateSizeGb:
    @patch("overflowml.inspect.model_estimator.inspect_model")
    def test_returns_fp16(self, mock_inspect):
        mock_inspect.return_value = ModelInfo(
            model_id="test", estimated_sizes_gb={"fp16": 14.0}, confidence="high",
        )
        assert estimate_size_gb("test") == 14.0

    @patch("overflowml.inspect.model_estimator.inspect_model")
    def test_fallback(self, mock_inspect):
        mock_inspect.return_value = ModelInfo(model_id="test")
        assert estimate_size_gb("test") == 14.0
