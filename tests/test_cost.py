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


def test_current_flagships():
    # Before adding these, Opus 4.8 / Sonnet 5 / GPT-5.5 had no pricing entry and
    # no prefix to fall back on, so estimate_cost returned None (un-costed batch).
    assert abs(estimate_cost("claude-opus-4-8", 1_000_000, 1_000_000) - 30.00) < 0.01
    assert abs(estimate_cost("claude-sonnet-5", 1_000_000, 1_000_000) - 18.00) < 0.01
    assert abs(estimate_cost("gpt-5.5", 1_000_000, 1_000_000) - 35.00) < 0.01
    assert abs(estimate_cost("gpt-5.4", 1_000_000, 1_000_000) - 17.50) < 0.01


def test_current_model_dated_prefix_match():
    # API ids carry a date suffix; the longest-prefix match must resolve them.
    cost = estimate_cost("claude-opus-4-8-20260528", 1_000_000, 0)
    assert cost is not None
    assert abs(cost - 5.00) < 0.01


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
