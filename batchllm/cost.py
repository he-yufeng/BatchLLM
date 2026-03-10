"""Token cost estimation for popular models."""

from __future__ import annotations

# prices per 1M tokens as of early 2026
# keeping it simple — users can override via config
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_1M, output_per_1M)
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o3-mini": (1.10, 4.40),
    # Anthropic
    "claude-opus-4-6": (15.00, 75.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    # Google
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    # DeepSeek
    "deepseek-chat": (0.14, 0.28),
    "deepseek-reasoner": (0.55, 2.19),
    # Mistral
    "mistral-large": (2.00, 6.00),
    "mistral-small": (0.20, 0.60),
}


def estimate_cost(
    model: str,
    tokens_in: int,
    tokens_out: int,
    custom_pricing: dict[str, tuple[float, float]] | None = None,
) -> float | None:
    """Estimate the cost in USD for a given token usage.

    Returns None if the model isn't in the pricing table.
    """
    pricing = {**MODEL_PRICING, **(custom_pricing or {})}

    # try exact match first, then longest prefix match
    prices = pricing.get(model)
    if prices is None:
        best_key = ""
        for key, val in pricing.items():
            if model.startswith(key) and len(key) > len(best_key):
                best_key = key
                prices = val

    if prices is None:
        return None

    input_cost = (tokens_in / 1_000_000) * prices[0]
    output_cost = (tokens_out / 1_000_000) * prices[1]
    return input_cost + output_cost


def format_cost(cost: float | None) -> str:
    """Format a cost as a human-readable string."""
    if cost is None:
        return "N/A (unknown model pricing)"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"
