"""Tests for the schema subset validator and the expect-schema pipeline."""

import csv
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from batchllm.processor import BatchConfig, BatchProcessor
from batchllm.schema import validate_schema


def _mock_response(content="test response", prompt_tokens=10, completion_tokens=20):
    """Build a mock OpenAI response (same shape as test_processor's)."""
    choice = MagicMock()
    choice.message.content = content
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def test_type_checks():
    assert validate_schema("x", {"type": "string"}) is None
    assert validate_schema(3, {"type": "string"}) == "$: expected type string, got int"
    assert validate_schema(True, {"type": "integer"}) == "$: expected type integer, got bool"
    assert validate_schema(3, {"type": "integer"}) is None
    assert validate_schema(3.5, {"type": "number"}) is None
    assert validate_schema(True, {"type": "boolean"}) is None
    assert validate_schema(None, {"type": "null"}) is None


def test_required_and_nested_properties():
    schema = {
        "type": "object",
        "required": ["name", "score"],
        "properties": {
            "name": {"type": "string"},
            "score": {"type": "number"},
            "meta": {
                "type": "object",
                "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
            },
        },
    }
    assert validate_schema({"name": "a", "score": 1}, schema) is None
    assert validate_schema({"name": "a"}, schema) == "$.score: missing required property"
    assert (
        validate_schema({"name": "a", "score": "x"}, schema)
        == "$.score: expected type number, got str"
    )
    assert (
        validate_schema({"name": "a", "score": 1, "meta": {"tags": ["ok", 2]}}, schema)
        == "$.meta.tags[1]: expected type string, got int"
    )


def test_enum_exact_type_match():
    schema = {"enum": ["red", "green"]}
    assert validate_schema("red", schema) is None
    assert "not in enum" in validate_schema("blue", schema)
    # 1 == True in Python; enum must not conflate them
    assert validate_schema(True, {"enum": [1]}) is not None


def test_unknown_keywords_are_ignored():
    assert validate_schema({"a": 1}, {"type": "object", "unknownKeyword": "anything"}) is None
    assert validate_schema(5, {"type": "mystery"}) is None


@pytest.mark.asyncio
async def test_schema_violation_retries_with_correction():

    config = BatchConfig(
        model="gpt-4o-mini",
        max_retries=1,
        expect_json=True,
        expect_schema={
            "type": "object",
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}},
        },
    )
    proc = BatchProcessor(config)
    mock_client = AsyncMock()
    # First reply fails the schema, the corrected retry passes.
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[_mock_response('{"wrong": 1}'), _mock_response('{"answer": "ok"}')]
    )
    proc._client = mock_client

    results = await proc.process_items(["hello"])

    assert len(results) == 1
    assert results[0].parsed_output == {"answer": "ok"}
    assert results[0].error is None
    # The retry carried the correction nudge with the violation spelled out.
    second_call_messages = mock_client.chat.completions.create.call_args_list[1].kwargs["messages"]
    assert any("missing required property" in m.get("content", "") for m in second_call_messages)


@pytest.mark.asyncio
async def test_schema_persistent_violation_records_failure():

    config = BatchConfig(
        model="gpt-4o-mini",
        max_retries=0,
        expect_json=True,
        expect_schema={"type": "object", "required": ["answer"]},
    )
    proc = BatchProcessor(config)
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_mock_response('{"wrong": 1}'))
    proc._client = mock_client

    results = await proc.process_items(["hello"])

    assert results[0].error is not None
    assert "schema violation" in results[0].error


def test_writer_adds_parsed_column_when_expect_json(tmp_path):
    proc = BatchProcessor(BatchConfig(model="gpt-4o-mini", expect_json=True))
    from batchllm.processor import BatchResult

    results = [
        BatchResult(index=0, input_text="a", output_text='{"x": 1}', parsed_output={"x": 1}),
    ]
    out = tmp_path / "out.csv"
    proc._write_output(out, results)
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["parsed"] == '{"x": 1}'

    out_jsonl = tmp_path / "out.jsonl"
    proc._write_output(out_jsonl, results)
    line = json.loads(out_jsonl.read_text(encoding="utf-8").strip())
    assert line["parsed"] == {"x": 1}


def test_writer_omits_parsed_column_without_expect_json(tmp_path):
    proc = BatchProcessor(BatchConfig(model="gpt-4o-mini"))
    from batchllm.processor import BatchResult

    out = tmp_path / "out.csv"
    proc._write_output(out, [BatchResult(index=0, input_text="a", output_text="plain")])
    with open(out, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert "parsed" not in reader.fieldnames
