"""Classify request failures into actionable categories.

A finished batch usually answers two questions: how many rows failed, and
why. The "why" is the part that tells you what to do next — back off and
rerun on a rate limit, fix your key on an auth error, or look at the input
on a 400. This module maps an exception to a short category so the processor
can aggregate it and the CLI can show a breakdown instead of one opaque count.
"""

from __future__ import annotations

import openai

RATE_LIMIT = "rate_limit"
AUTH = "auth"
TIMEOUT = "timeout"
CONNECTION = "connection"
BAD_REQUEST = "bad_request"
CONFLICT = "conflict"
SERVER = "server"
OTHER = "other"
UNKNOWN = "unknown"

# Human-readable labels for the summary table. Keep them short; the category
# key is what gets stored in checkpoints and stats.
LABELS: dict[str, str] = {
    RATE_LIMIT: "Rate limit (429)",
    AUTH: "Auth / permission",
    TIMEOUT: "Timeout",
    CONNECTION: "Connection",
    BAD_REQUEST: "Bad request (4xx)",
    CONFLICT: "Conflict (409)",
    SERVER: "Server error (5xx)",
    OTHER: "Other",
    UNKNOWN: "Unknown (legacy checkpoint)",
}

# (exception class, category), most specific first so isinstance picks the
# tightest match. APITimeoutError subclasses APIConnectionError, so it has to
# come first. Classes are looked up by name to stay resilient across openai
# releases — a renamed or dropped class just falls through to the heuristics.
_TYPE_CATEGORIES: list[tuple[type, str]] = []
for _name, _category in [
    ("RateLimitError", RATE_LIMIT),
    ("AuthenticationError", AUTH),
    ("PermissionDeniedError", AUTH),
    ("APITimeoutError", TIMEOUT),
    ("APIConnectionError", CONNECTION),
    ("BadRequestError", BAD_REQUEST),
    ("NotFoundError", BAD_REQUEST),
    ("UnprocessableEntityError", BAD_REQUEST),
    ("ConflictError", CONFLICT),
    ("InternalServerError", SERVER),
]:
    _cls = getattr(openai, _name, None)
    if isinstance(_cls, type) and issubclass(_cls, BaseException):
        _TYPE_CATEGORIES.append((_cls, _category))


def _from_status_code(status: int) -> str | None:
    if status == 429:
        return RATE_LIMIT
    if status in (401, 403):
        return AUTH
    if status == 409:
        return CONFLICT
    if 400 <= status < 500:
        return BAD_REQUEST
    if status >= 500:
        return SERVER
    return None


def classify_error(exc: BaseException) -> str:
    """Map an exception to a failure category.

    Tries the known openai exception types first, then any HTTP status code
    carried on the exception, then the builtin timeout/connection errors, and
    finally falls back to ``other``.
    """
    for exc_type, category in _TYPE_CATEGORIES:
        if isinstance(exc, exc_type):
            return category

    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        category = _from_status_code(status)
        if category is not None:
            return category

    if isinstance(exc, TimeoutError):
        return TIMEOUT
    if isinstance(exc, ConnectionError):
        return CONNECTION

    return OTHER


def describe(category: str) -> str:
    """Return a human-readable label for a category key."""
    return LABELS.get(category, category)
