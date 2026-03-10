"""Tests for cost estimation."""

from batchllm.cost import estimate_cost, format_cost


def test_known_model():
    cost = estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    assert cost is not None
    assert abs(cost - 0.75) < 0.01  # 0.15 + 0.60


def test_unknown_model():
    cost = estimate_cost("some-random-model-xyz", 1000, 1000)
    assert cost is None


def test_prefix_match():
    cost = estimate_cost("claude-3-haiku-20240307", 1_000_000, 0)
    assert cost is not None
    assert abs(cost - 0.25) < 0.01


def test_custom_pricing():
    custom = {"my-model": (1.0, 2.0)}
    cost = estimate_cost("my-model", 1_000_000, 1_000_000, custom_pricing=custom)
    assert cost is not None
    assert abs(cost - 3.0) < 0.01


def test_format_cost_none():
    assert "N/A" in format_cost(None)


def test_format_cost_small():
    assert format_cost(0.001) == "$0.0010"


def test_format_cost_normal():
    assert format_cost(1.50) == "$1.50"
