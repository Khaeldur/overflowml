"""Tests for model inspection and size estimation."""

import json
import struct
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from overflowml.inspect.arch_registry import classify_task, estimate_params_from_config
from overflowml.inspect.hf_probe import ModelSizeUnknown, WeightsProbe, probe_safetensors_params
from overflowml.inspect.model_estimator import (
    estimate_size_gb, inspect_model, is_prequantized, planning_size_gb, size_label, weights_bytes,
)
from overflowml.core.types import ModelInfo

GIB = 1024 ** 3
EST = "overflowml.inspect.model_estimator"

# Real Hub metadata. `parameters` (model_info API) counts packed quantized weights as
# unpacked params; `stored` (safetensors headers) has the real stored element counts.
QWEN25_7B_PARAMS = {"BF16": 7_615_616_512}
BNB_4BIT = {  # unsloth/Llama-3.2-1B-Instruct-bnb-4bit, 0.96 GiB of files
    "parameters": {"F32": 89_856, "BF16": 262_735_872, "U8": 1_003_525_730},
    "stored": {"F32": 89_856, "BF16": 262_735_872, "U8": 501_762_865},
}
AWQ_7B = {  # Qwen/Qwen2.5-7B-Instruct-AWQ, 5.19 GiB of files
    "parameters": {"I32": 6_525_288_448, "F16": 1_090_328_064},
    "stored": {"I32": 822_033_408, "F16": 1_141_306_880},
}

# Real config.json fields (only what the estimator reads)
LLAMA3_8B_CONFIG = {
    "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
    "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 128256,
    "tie_word_embeddings": False,
}
QWEN25_05B_CONFIG = {
    "hidden_size": 896, "num_hidden_layers": 24, "num_attention_heads": 14,
    "num_key_value_heads": 2, "intermediate_size": 4864, "vocab_size": 151936,
    "tie_word_embeddings": True,
}
QWEN3_30B_A3B_CONFIG = {
    "hidden_size": 2048, "num_hidden_layers": 48, "num_attention_heads": 32,
    "num_key_value_heads": 4, "head_dim": 128, "intermediate_size": 6144,
    "moe_intermediate_size": 768, "num_experts": 128, "vocab_size": 151936,
    "tie_word_embeddings": False,
}


def _hub_error(cls):
    # huggingface_hub HTTP errors require a response object; the probe only checks the type
    return cls.__new__(cls)


def _write_safetensors(path, tensors):
    """Write a header-only safetensors file: {name: (dtype, shape)}."""
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset]}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


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

    @pytest.mark.parametrize("config, real_params", [
        (LLAMA3_8B_CONFIG, 8_030_261_248),        # GQA
        (QWEN25_05B_CONFIG, 494_032_768),         # tied embeddings
        (QWEN3_30B_A3B_CONFIG, 30_532_122_624),   # MoE with moe_intermediate_size (was ~233B)
    ])
    def test_matches_real_param_counts(self, config, real_params):
        count, _ = estimate_params_from_config(config)
        assert abs(count - real_params) / real_params < 0.03

    def test_nested_text_config(self):
        count, _ = estimate_params_from_config({"architectures": ["SomeVLM"], "text_config": LLAMA3_8B_CONFIG})
        assert abs(count - 8_030_261_248) / 8_030_261_248 < 0.03


class TestWeightsBytes:
    def test_bf16_is_two_bytes(self):
        assert weights_bytes(QWEN25_7B_PARAMS) == 2 * 7_615_616_512

    def test_mixed_dtypes(self):
        assert weights_bytes({"F32": 10, "BF16": 10, "U8": 10, "F4": 10}) == 40 + 20 + 10 + 5

    @pytest.mark.parametrize("stored, real_gib", [(BNB_4BIT["stored"], 0.96), (AWQ_7B["stored"], 5.19)])
    def test_stored_counts_match_file_sizes(self, stored, real_gib):
        assert weights_bytes(stored) / GIB == pytest.approx(real_gib, abs=0.01)

    def test_prequantized_detection(self):
        assert is_prequantized(BNB_4BIT["parameters"])
        assert is_prequantized(AWQ_7B["parameters"])
        assert not is_prequantized(QWEN25_7B_PARAMS)
        # int64 position_ids buffers in a plain fp16 checkpoint are not quantization
        assert not is_prequantized({"F16": 1_000_000, "I64": 512})


class TestProbeSafetensorsParams:
    def test_returns_parameter_counts(self):
        info = SimpleNamespace(
            safetensors=SimpleNamespace(parameters=QWEN25_7B_PARAMS, total=7_615_616_512),
            siblings=[SimpleNamespace(rfilename="config.json")],
        )
        with patch("huggingface_hub.model_info", return_value=info):
            probe = probe_safetensors_params("Qwen/Qwen2.5-7B-Instruct")
        assert probe.params_by_dtype == QWEN25_7B_PARAMS
        assert probe.pipeline is False

    def test_diffusers_repo_flagged_as_pipeline(self):
        info = SimpleNamespace(
            safetensors=SimpleNamespace(parameters={"BF16": 20_430_401_088}, total=20_430_401_088),
            siblings=[SimpleNamespace(rfilename="model_index.json")],
        )
        with patch("huggingface_hub.model_info", return_value=info):
            assert probe_safetensors_params("Qwen/Qwen-Image-Edit-2509").pipeline is True

    def test_no_metadata_returns_none(self):
        with patch("huggingface_hub.model_info", return_value=SimpleNamespace(safetensors=None)):
            assert probe_safetensors_params("org/pytorch-bin-only") is None

    def test_not_found(self):
        from huggingface_hub.utils import RepositoryNotFoundError
        with patch("huggingface_hub.model_info", side_effect=_hub_error(RepositoryNotFoundError)):
            with pytest.raises(ModelSizeUnknown) as exc:
                probe_safetensors_params("typo-org/nope")
        assert exc.value.reason == "not_found"

    def test_gated(self):
        from huggingface_hub.utils import GatedRepoError
        with patch("huggingface_hub.model_info", side_effect=_hub_error(GatedRepoError)):
            with pytest.raises(ModelSizeUnknown) as exc:
                probe_safetensors_params("org/gated")
        assert exc.value.reason == "gated"

    def test_network_error_is_offline(self):
        with patch("huggingface_hub.model_info", side_effect=ConnectionError("no route")):
            with pytest.raises(ModelSizeUnknown) as exc:
                probe_safetensors_params("org/model")
        assert exc.value.reason == "offline"


class TestLocalDirectory:
    def test_reads_headers(self, tmp_path):
        _write_safetensors(tmp_path / "model.safetensors", {"w": ("BF16", [10, 20]), "b": ("F32", [20])})
        assert probe_safetensors_params(str(tmp_path)).params_by_dtype == {"BF16": 200, "F32": 20}

    def test_index_limits_files(self, tmp_path):
        # Mistral-style: HF shards + an index, plus a consolidated copy of the same weights
        _write_safetensors(tmp_path / "model-00001-of-00002.safetensors", {"a": ("BF16", [100])})
        _write_safetensors(tmp_path / "model-00002-of-00002.safetensors", {"b": ("BF16", [100])})
        _write_safetensors(tmp_path / "consolidated.safetensors", {"a": ("BF16", [100]), "b": ("BF16", [100])})
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            "a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors",
        }}))
        assert probe_safetensors_params(str(tmp_path)).params_by_dtype == {"BF16": 200}

    def test_consolidated_skipped_without_index(self, tmp_path):
        _write_safetensors(tmp_path / "model.safetensors", {"a": ("BF16", [100])})
        _write_safetensors(tmp_path / "consolidated.safetensors", {"a": ("BF16", [100])})
        assert probe_safetensors_params(str(tmp_path)).params_by_dtype == {"BF16": 100}

    def test_git_lfs_pointer_is_size_unknown(self, tmp_path):
        (tmp_path / "model.safetensors").write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n"
        )
        with pytest.raises(ModelSizeUnknown) as exc:
            probe_safetensors_params(str(tmp_path))
        assert exc.value.reason == "no_metadata"

    def test_local_file_is_size_unknown(self, tmp_path):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"GGUF")
        with pytest.raises(ModelSizeUnknown) as exc:
            probe_safetensors_params(str(f))
        assert "local file" in str(exc.value)

    def test_local_dirs_bypass_cache(self, tmp_path, monkeypatch):
        from overflowml.core import cache
        monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
        model = tmp_path / "m"
        model.mkdir()
        _write_safetensors(model / "model.safetensors", {"w": ("BF16", [1_000_000])})
        first = inspect_model(str(model))
        _write_safetensors(model / "model.safetensors", {"w": ("BF16", [8_000_000])})
        assert inspect_model(str(model)).param_count == 8 * first.param_count


class TestInspectModel:
    @patch(f"{EST}.probe_safetensors_params", return_value=WeightsProbe(QWEN25_7B_PARAMS))
    @patch(f"{EST}.probe_config", return_value={
        "architectures": ["Qwen2ForCausalLM"], "hidden_size": 3584,
    })
    def test_safetensors_path(self, mock_config, mock_st):
        info = inspect_model("Qwen/Qwen2.5-7B-Instruct", use_cache=False)
        assert info.confidence == "high"
        assert info.source == "safetensors metadata"
        assert info.architecture == "Qwen2ForCausalLM"

    @patch(f"{EST}.probe_safetensors_params", return_value=WeightsProbe(QWEN25_7B_PARAMS))
    @patch(f"{EST}.probe_config", return_value=None)
    def test_hub_total_is_params_not_bytes(self, mock_config, mock_st):
        # Regression: the Hub total (7.6B params) used to be halved as if it were bytes
        info = inspect_model("Qwen/Qwen2.5-7B-Instruct", use_cache=False)
        assert info.param_count == 7_615_616_512
        assert info.estimated_sizes_gb["fp16"] == pytest.approx(14.19, abs=0.01)
        assert info.prequantized is False

    @pytest.mark.parametrize("repo, real_gib", [(BNB_4BIT, 0.96), (AWQ_7B, 5.19)])
    def test_prequantized_sized_from_stored_headers(self, repo, real_gib):
        # Regression: Hub `parameters` x container width overstated AWQ 5x, bnb 1.5x
        with patch(f"{EST}.probe_safetensors_params", return_value=WeightsProbe(repo["parameters"])), \
             patch(f"{EST}.probe_stored_params", return_value=repo["stored"]), \
             patch(f"{EST}.probe_config", return_value=None):
            info = inspect_model("org/quantized", use_cache=False)
        assert info.prequantized is True
        assert "fp16" not in info.estimated_sizes_gb
        assert planning_size_gb(info) == pytest.approx(real_gib, abs=0.01)
        assert size_label(info) == "as stored"

    def test_prequantized_without_headers_is_low_confidence(self):
        with patch(f"{EST}.probe_safetensors_params", return_value=WeightsProbe(AWQ_7B["parameters"])), \
             patch(f"{EST}.probe_stored_params", return_value=None), \
             patch(f"{EST}.probe_config", return_value=None):
            info = inspect_model("org/quantized", use_cache=False)
        assert info.confidence == "low"

    @patch(f"{EST}.probe_safetensors_params", return_value=WeightsProbe({"BF16": 20_430_401_088}, pipeline=True))
    @patch(f"{EST}.probe_config", return_value=None)
    def test_diffusers_pipeline_is_flagged(self, mock_config, mock_st):
        info = inspect_model("Qwen/Qwen-Image-Edit-2509", use_cache=False)
        assert info.task_family == "diffusers"
        assert info.confidence == "medium"
        assert any("main component" in n for n in info.notes)

    @patch(f"{EST}.probe_safetensors_params", return_value=None)
    @patch(f"{EST}.probe_config", return_value={
        "architectures": ["MistralForCausalLM"],
        "hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 32000,
    })
    def test_config_fallback(self, mock_config, mock_st):
        info = inspect_model("test/model-cfg", use_cache=False)
        assert info.confidence == "medium"
        assert "config.json" in info.source
        assert info.param_count > 0

    @patch(f"{EST}.probe_safetensors_params", return_value=None)
    @patch(f"{EST}.probe_config", return_value={**LLAMA3_8B_CONFIG, "quantization_config": {"bits": 4}})
    def test_quantized_config_without_metadata_raises(self, mock_config, mock_st):
        # An fp16 estimate of a GPTQ .bin checkpoint would be ~4x too large
        with pytest.raises(ModelSizeUnknown):
            inspect_model("TheBloke/some-gptq-bin", use_cache=False)

    @patch(f"{EST}.probe_safetensors_params", return_value=None)
    @patch(f"{EST}.probe_config", return_value=None)
    def test_no_data(self, mock_config, mock_st):
        info = inspect_model("test/nonexistent", use_cache=False)
        assert info.confidence == "low"
        assert info.param_count is None

    @patch(f"{EST}.probe_safetensors_params", side_effect=ModelSizeUnknown("typo-org/nope", "not_found"))
    def test_not_found_raises(self, mock_st):
        with pytest.raises(ModelSizeUnknown):
            inspect_model("typo-org/nope", use_cache=False)

    @patch(f"{EST}.probe_safetensors_params", side_effect=ModelSizeUnknown("org/m", "offline"))
    @patch(f"{EST}.probe_config", return_value=LLAMA3_8B_CONFIG)
    def test_offline_falls_back_to_cached_config(self, mock_config, mock_st):
        info = inspect_model("org/m", use_cache=False)
        assert info.confidence == "medium"

    @patch(f"{EST}.probe_safetensors_params", side_effect=ModelSizeUnknown("org/m", "offline"))
    @patch(f"{EST}.probe_config", return_value=None)
    def test_offline_without_config_raises(self, mock_config, mock_st):
        with pytest.raises(ModelSizeUnknown) as exc:
            inspect_model("org/m", use_cache=False)
        assert exc.value.reason == "offline"

    @patch(f"{EST}.probe_safetensors_params", side_effect=ModelSizeUnknown("org/m", "offline"))
    @patch(f"{EST}.probe_config", return_value={"architectures": ["X"]})
    def test_offline_with_useless_config_raises_offline(self, mock_config, mock_st):
        with pytest.raises(ModelSizeUnknown) as exc:
            inspect_model("org/m", use_cache=False)
        assert exc.value.reason == "offline"

    def test_offline_estimate_is_not_cached(self, tmp_path, monkeypatch):
        from overflowml.core import cache
        monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
        with patch(f"{EST}.probe_safetensors_params", side_effect=ModelSizeUnknown("org/m", "offline")), \
             patch(f"{EST}.probe_config", return_value=LLAMA3_8B_CONFIG):
            inspect_model("org/m")
        assert cache.load_cached_model("org/m") is None


class TestEstimateSizeGb:
    @patch(f"{EST}.inspect_model")
    def test_returns_fp16(self, mock_inspect):
        mock_inspect.return_value = ModelInfo(
            model_id="test", estimated_sizes_gb={"fp16": 14.0}, confidence="high",
        )
        assert estimate_size_gb("test") == 14.0

    @patch(f"{EST}.inspect_model")
    def test_unknown_size_raises(self, mock_inspect):
        # Used to silently return 14.0
        mock_inspect.return_value = ModelInfo(model_id="test")
        with pytest.raises(ModelSizeUnknown) as exc:
            estimate_size_gb("test")
        assert exc.value.reason == "no_metadata"

    def test_model_size_unknown_is_value_error(self):
        assert issubclass(ModelSizeUnknown, ValueError)


class TestModelCacheSchema:
    def test_entries_without_schema_are_ignored(self, tmp_path, monkeypatch):
        from overflowml.core import cache
        monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
        cache.save_cached_model("org/m", {"model_id": "org/m"})
        assert cache.load_cached_model("org/m") is not None

        # An entry written before the sizing fix (no _schema) must not be served
        path = next(tmp_path.glob("model_*.json"))
        data = json.loads(path.read_text())
        del data["_schema"]
        path.write_text(json.dumps(data))
        assert cache.load_cached_model("org/m") is None
