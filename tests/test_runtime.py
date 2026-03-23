"""Tests for runtime intelligence — KV cache, flash attention, PCIe, fragmentation, precision."""

from overflowml.core.runtime import (
    KVCacheEstimate,
    FlashAttentionStatus,
    PCIeBandwidth,
    FragmentationReport,
    LayerPrecisionPlan,
    LoadTimeEstimate,
    estimate_kv_cache,
    estimate_kv_cache_from_config,
    detect_flash_attention,
    detect_pcie_bandwidth,
    diagnose_fragmentation,
    plan_layer_precision,
    context_adjusted_vram,
    estimate_load_time,
    suggest_draft_model,
    estimate_prefix_savings,
    estimate_transfer_overhead,
)


class TestKVCache:
    def test_basic_estimate(self):
        kv = estimate_kv_cache(num_layers=32, num_kv_heads=32, head_dim=128, context_length=4096)
        assert isinstance(kv, KVCacheEstimate)
        assert kv.cache_gb > 0
        assert kv.per_token_mb > 0

    def test_longer_context_more_cache(self):
        kv_short = estimate_kv_cache(context_length=2048)
        kv_long = estimate_kv_cache(context_length=32768)
        assert kv_long.cache_gb > kv_short.cache_gb

    def test_gqa_reduces_cache(self):
        kv_mha = estimate_kv_cache(num_kv_heads=32)  # multi-head
        kv_gqa = estimate_kv_cache(num_kv_heads=8)   # grouped-query
        assert kv_gqa.cache_gb < kv_mha.cache_gb

    def test_batch_multiplies(self):
        kv_1 = estimate_kv_cache(batch_size=1)
        kv_4 = estimate_kv_cache(batch_size=4)
        assert abs(kv_4.cache_gb - kv_1.cache_gb * 4) < 0.01

    def test_from_config(self):
        config = {"hidden_size": 4096, "num_hidden_layers": 32,
                  "num_attention_heads": 32, "num_key_value_heads": 8}
        kv = estimate_kv_cache_from_config(config, context_length=4096)
        assert kv.num_kv_heads == 8  # GQA detected


class TestContextAdjustedVram:
    def test_includes_all_components(self):
        result = context_adjusted_vram(model_weights_gb=14.0, context_length=4096)
        assert result["weights_gb"] == 14.0
        assert result["kv_cache_gb"] > 0
        assert result["activation_gb"] > 0
        assert result["total_gb"] > 14.0

    def test_longer_context_more_total(self):
        r_short = context_adjusted_vram(14.0, context_length=2048)
        r_long = context_adjusted_vram(14.0, context_length=32768)
        assert r_long["total_gb"] > r_short["total_gb"]


class TestFlashAttention:
    def test_returns_status(self):
        status = detect_flash_attention()
        assert isinstance(status, FlashAttentionStatus)
        assert len(status.notes) > 0

    def test_has_backend_field(self):
        status = detect_flash_attention()
        assert status.backend in ("flash_attn", "sdpa", "xformers", "none")


class TestPCIe:
    def test_returns_bandwidth(self):
        bw = detect_pcie_bandwidth()
        assert isinstance(bw, PCIeBandwidth)

    def test_transfer_estimate(self):
        result = estimate_transfer_overhead(
            model_ram_gb=50.0,
            pcie=PCIeBandwidth(generation=4, width=16, practical_gbps=22.0, detected=True),
        )
        assert result["transfer_time_s"] > 0
        assert "50GB" in result["note"]


class TestFragmentation:
    def test_returns_report(self):
        report = diagnose_fragmentation()
        assert isinstance(report, FragmentationReport)
        assert len(report.notes) > 0


class TestLayerPrecision:
    def test_basic_plan(self):
        plan = plan_layer_precision(num_layers=32, hidden_size=4096, vocab_size=32000)
        assert isinstance(plan, LayerPrecisionPlan)
        assert len(plan.layers) > 0
        assert plan.total_optimized_gb > 0

    def test_savings(self):
        plan = plan_layer_precision(num_layers=32, hidden_size=4096, vocab_size=32000)
        assert plan.savings_pct > 0  # mixed precision should save something

    def test_critical_layers_fp16(self):
        plan = plan_layer_precision(num_layers=32)
        embed = [l for l in plan.layers if l["name"] == "embedding"][0]
        assert embed["precision"] == "fp16"
        head = [l for l in plan.layers if l["name"] == "lm_head"][0]
        assert head["precision"] == "fp16"

    def test_middle_layers_quantized(self):
        plan = plan_layer_precision(num_layers=32)
        middle_ffn = [l for l in plan.layers if "layer_16_ffn" in l["name"]][0]
        assert middle_ffn["precision"] == "int4"


class TestLoadTime:
    def test_basic(self):
        est = estimate_load_time(14.0)
        assert isinstance(est, LoadTimeEstimate)
        assert est.estimated_seconds > 0

    def test_mmap_faster(self):
        normal = estimate_load_time(100.0, use_mmap=False)
        mmap = estimate_load_time(100.0, use_mmap=True)
        assert mmap.estimated_seconds < normal.estimated_seconds

    def test_large_model_hint(self):
        est = estimate_load_time(100.0)
        assert any("safetensors" in n for n in est.notes)


class TestSpeculativeDecode:
    def test_known_model(self):
        result = suggest_draft_model("meta-llama/Llama-3-70B")
        assert result is not None
        assert "draft_model" in result
        assert result["expected_speedup"] == "2-3x"

    def test_unknown_model(self):
        result = suggest_draft_model("totally-unknown/model-xyz")
        assert result is None

    def test_mixtral(self):
        result = suggest_draft_model("mistralai/Mixtral-8x7B")
        assert result is not None


class TestPrefixSavings:
    def test_basic(self):
        result = estimate_prefix_savings(
            num_requests=10,
            shared_prefix_tokens=1000,
            total_tokens_per_request=2000,
        )
        assert result["savings_gb"] > 0
        assert result["savings_pct"] > 0

    def test_no_sharing(self):
        result = estimate_prefix_savings(
            num_requests=10,
            shared_prefix_tokens=0,
            total_tokens_per_request=2000,
        )
        assert result["savings_gb"] == 0
