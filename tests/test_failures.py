"""Tests for failure classification."""

import httpx
import openai
import pytest

from batchllm.failures import (
    AUTH,
    BAD_REQUEST,
    CONNECTION,
    OTHER,
    RATE_LIMIT,
    SERVER,
    TIMEOUT,
    classify_error,
    describe,
)


def _openai_status_error(cls, status: int):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return cls("boom", response=response, body=None)


class TestClassifyOpenAIErrors:
    def test_rate_limit(self):
        exc = _openai_status_error(openai.RateLimitError, 429)
        assert classify_error(exc) == RATE_LIMIT

    def test_auth(self):
        exc = _openai_status_error(openai.AuthenticationError, 401)
        assert classify_error(exc) == AUTH

    def test_permission_is_auth(self):
        exc = _openai_status_error(openai.PermissionDeniedError, 403)
        assert classify_error(exc) == AUTH

    def test_bad_request(self):
        exc = _openai_status_error(openai.BadRequestError, 400)
        assert classify_error(exc) == BAD_REQUEST

    def test_timeout(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        assert classify_error(openai.APITimeoutError(request)) == TIMEOUT

    def test_connection(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        exc = openai.APIConnectionError(message="down", request=request)
        assert classify_error(exc) == CONNECTION


class TestClassifyFallbacks:
    def test_status_code_attribute(self):
        class Weird(Exception):
            status_code = 503

        assert classify_error(Weird()) == SERVER

    def test_builtin_timeout(self):
        assert classify_error(TimeoutError("slow")) == TIMEOUT

    def test_builtin_connection(self):
        assert classify_error(ConnectionError("refused")) == CONNECTION

    def test_unknown_falls_back_to_other(self):
        assert classify_error(ValueError("nope")) == OTHER


def test_describe_known_and_unknown():
    assert describe(RATE_LIMIT) == "Rate limit (429)"
    assert describe("made_up_category") == "made_up_category"


@pytest.mark.parametrize("category", [RATE_LIMIT, AUTH, TIMEOUT, CONNECTION, BAD_REQUEST])
def test_every_real_category_has_a_label(category):
    assert describe(category) != category
