"""Tests for auto-batching engine."""

from unittest.mock import MagicMock, patch

from overflowml.batch import (
    BatchConfig,
    auto_batch,
    calculate_batch_size,
    measure_vram_headroom,
)


class TestMeasureVramHeadroom:
    def test_returns_tuple(self):
        available, total = measure_vram_headroom()
        assert isinstance(available, float)
        assert isinstance(total, float)
        assert available >= 0
        assert total >= 0


class TestCalculateBatchSize:
    def test_basic(self):
        config = calculate_batch_size(
            available_vram_gb=10.0,
            per_item_gb=2.0,
        )
        assert isinstance(config, BatchConfig)
        assert config.batch_size == 4  # 10 * 0.85 / 2 = 4.25 → 4

    def test_large_headroom(self):
        config = calculate_batch_size(
            available_vram_gb=24.0,
            per_item_gb=3.0,
        )
        # 24 * 0.85 / 3 = 6.8 → 6
        assert config.batch_size == 6

    def test_tiny_headroom(self):
        config = calculate_batch_size(
            available_vram_gb=0.5,
            per_item_gb=3.0,
        )
        assert config.batch_size == 1  # min

    def test_zero_headroom(self):
        config = calculate_batch_size(
            available_vram_gb=0.0,
            per_item_gb=1.0,
        )
        assert config.batch_size == 1

    def test_max_batch_size_cap(self):
        config = calculate_batch_size(
            available_vram_gb=1000.0,
            per_item_gb=0.1,
            max_batch_size=16,
        )
        assert config.batch_size == 16

    def test_safety_margin(self):
        config = calculate_batch_size(
            available_vram_gb=10.0,
            per_item_gb=1.0,
            safety_margin=0.5,
        )
        # 10 * 0.5 / 1 = 5
        assert config.batch_size == 5

    def test_notes_populated(self):
        config = calculate_batch_size(
            available_vram_gb=10.0,
            per_item_gb=2.0,
        )
        assert len(config.notes) > 0
        assert "batch_size" in config.notes[-1].lower() or "optimal" in config.notes[-1].lower()


class TestAutoBatch:
    def test_basic_batching(self):
        items = list(range(10))
        batches = list(auto_batch(items, batch_size=3))
        assert len(batches) == 4  # [3, 3, 3, 1]
        assert batches[0] == [0, 1, 2]
        assert batches[-1] == [9]

    def test_batch_size_1(self):
        items = ["a", "b", "c"]
        batches = list(auto_batch(items, batch_size=1))
        assert len(batches) == 3
        assert all(len(b) == 1 for b in batches)

    def test_exact_division(self):
        items = list(range(9))
        batches = list(auto_batch(items, batch_size=3))
        assert len(batches) == 3
        assert all(len(b) == 3 for b in batches)

    def test_auto_calculation(self):
        items = list(range(20))
        batches = list(auto_batch(
            items,
            per_item_gb=1.0,
            batch_size=None,  # auto
        ))
        assert len(batches) > 0
        # Should produce at least 1 batch
        total = sum(len(b) for b in batches)
        assert total == 20

    def test_empty_items(self):
        batches = list(auto_batch([], batch_size=4))
        assert len(batches) == 0


class TestBatchConfig:
    def test_defaults(self):
        c = BatchConfig()
        assert c.batch_size == 1
        assert c.safety_margin == 0.85
        assert c.notes is not None

    def test_with_values(self):
        c = BatchConfig(
            batch_size=4,
            available_vram_gb=20.0,
            estimated_per_item_gb=3.0,
            method="architecture_estimate",
        )
        assert c.batch_size == 4
