"""Tests for the batch processor."""

import csv
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from batchllm.processor import BatchConfig, BatchProcessor, BatchResult


@pytest.fixture
def config():
    return BatchConfig(
        model="gpt-4o-mini",
        max_concurrent=2,
        max_retries=1,
    )


@pytest.fixture
def processor(config):
    return BatchProcessor(config)


def _mock_response(content="test response", prompt_tokens=10, completion_tokens=20):
    """Build a mock OpenAI response."""
    choice = MagicMock()
    choice.message.content = content

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


class TestBatchConfig:
    def test_defaults(self):
        c = BatchConfig()
        assert c.model == "gpt-4o-mini"
        assert c.max_concurrent == 10
        assert c.max_retries == 3
        assert c.prompt_template == "{input}"

    def test_custom(self):
        c = BatchConfig(model="gpt-4o", max_concurrent=5)
        assert c.model == "gpt-4o"
        assert c.max_concurrent == 5


class TestBuildMessages:
    def test_no_system(self, processor):
        msgs = processor._build_messages("hello")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "hello"

    def test_with_system(self):
        config = BatchConfig(system_prompt="You are helpful")
        proc = BatchProcessor(config)
        msgs = proc._build_messages("hello")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are helpful"

    def test_template(self):
        config = BatchConfig(prompt_template="Translate: {input}")
        proc = BatchProcessor(config)
        msgs = proc._build_messages("hello world")
        assert msgs[0]["content"] == "Translate: hello world"

    def test_template_substitutes_extra_fields(self):
        config = BatchConfig(prompt_template="Translate to {language}: {input}")
        proc = BatchProcessor(config)
        msgs = proc._build_messages("hello", {"input": "hello", "language": "French"})
        assert msgs[0]["content"] == "Translate to French: hello"

    def test_template_leaves_unknown_placeholder_literal(self):
        config = BatchConfig(prompt_template="{input} [{missing}]")
        proc = BatchProcessor(config)
        msgs = proc._build_messages("hi", {"input": "hi"})
        assert msgs[0]["content"] == "hi [{missing}]"

    def test_template_does_not_re_expand_substituted_values(self):
        # an input that contains a brace token must not be expanded again
        config = BatchConfig(prompt_template="{input} / {label}")
        proc = BatchProcessor(config)
        msgs = proc._build_messages("{label}", {"input": "{label}", "label": "x"})
        assert msgs[0]["content"] == "{label} / x"

    def test_template_without_fields_is_backward_compatible(self):
        config = BatchConfig(prompt_template="Q: {input}")
        proc = BatchProcessor(config)
        assert proc._build_messages("hi")[0]["content"] == "Q: hi"


class TestReadInput:
    def test_read_csv(self, processor, tmp_path):
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["input", "label"])
            writer.writeheader()
            writer.writerow({"input": "hello", "label": "greeting"})
            writer.writerow({"input": "bye", "label": "farewell"})

        items, fields = processor._read_input(csv_file)
        assert items == ["hello", "bye"]
        # the full row is preserved so other columns can fill template placeholders
        assert fields == [
            {"input": "hello", "label": "greeting"},
            {"input": "bye", "label": "farewell"},
        ]

    def test_read_jsonl(self, processor, tmp_path):
        jsonl_file = tmp_path / "data.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(json.dumps({"input": "hello"}) + "\n")
            f.write(json.dumps({"input": "world"}) + "\n")

        items, _ = processor._read_input(jsonl_file)
        assert items == ["hello", "world"]

    def test_read_jsonl_text_fallback(self, processor, tmp_path):
        jsonl_file = tmp_path / "data.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(json.dumps({"text": "hello"}) + "\n")

        items, _ = processor._read_input(jsonl_file)
        assert items == ["hello"]

    def test_read_jsonl_missing_input_fails(self, processor, tmp_path):
        jsonl_file = tmp_path / "data.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(json.dumps({"question": "hello"}) + "\n")

        with pytest.raises(ValueError, match="missing 'input'"):
            processor._read_input(jsonl_file)

    def test_read_plain_text(self, processor, tmp_path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("line one\nline two\n\nline three\n")

        items, _ = processor._read_input(txt_file)
        assert items == ["line one", "line two", "line three"]

    def test_read_jsonl_strings(self, processor, tmp_path):
        jsonl_file = tmp_path / "data.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(json.dumps("hello") + "\n")
            f.write(json.dumps("world") + "\n")

        items, _ = processor._read_input(jsonl_file)
        assert items == ["hello", "world"]

    def test_read_csv_missing_input_column_fails(self, processor, tmp_path):
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["question"])
            writer.writeheader()
            writer.writerow({"question": "hello"})

        with pytest.raises(ValueError, match="missing input column 'input'"):
            processor._read_input(csv_file)


class TestWriteOutput:
    def test_write_csv(self, processor, tmp_path):
        results = [
            BatchResult(
                index=0,
                input_text="hi",
                output_text="hello",
                tokens_in=5,
                tokens_out=5,
            ),
            BatchResult(
                index=1,
                input_text="bye",
                output_text="goodbye",
                tokens_in=5,
                tokens_out=7,
            ),
        ]
        out = tmp_path / "out.csv"
        processor._write_output(out, results)

        with open(out, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["input"] == "hi"
        assert rows[0]["output"] == "hello"

    def test_write_jsonl(self, processor, tmp_path):
        results = [
            BatchResult(index=0, input_text="hi", output_text="hello", tokens_in=5, tokens_out=5),
        ]
        out = tmp_path / "out.jsonl"
        processor._write_output(out, results)

        with open(out) as f:
            data = json.loads(f.readline())
        assert data["input"] == "hi"
        assert data["output"] == "hello"


class TestCheckpoint:
    def test_save_and_load(self, processor, tmp_path):
        ckpt = tmp_path / "checkpoint.jsonl"
        processor._checkpoint_path = ckpt

        result = BatchResult(
            index=0, input_text="test", output_text="response", tokens_in=10, tokens_out=20
        )
        processor._save_checkpoint(result)

        # new processor, load checkpoint
        new_proc = BatchProcessor()
        new_proc._load_checkpoint(ckpt)
        assert 0 in new_proc._completed_indices
        assert len(new_proc._results) == 1
        assert new_proc._results[0].output_text == "response"

    def test_save_creates_checkpoint_parent(self, processor, tmp_path):
        ckpt = tmp_path / "nested" / "checkpoint.jsonl"
        processor._checkpoint_path = ckpt
        processor._save_checkpoint(BatchResult(index=0, input_text="test", output_text="ok"))

        assert ckpt.exists()


@pytest.mark.asyncio
async def test_process_items_mock(config):
    """Test the full pipeline with a mocked API."""
    proc = BatchProcessor(config)
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_mock_response("done"))
    proc._client = mock_client

    results = await proc.process_items(["hello", "world"])

    assert len(results) == 2
    assert all(r.output_text == "done" for r in results)
    assert proc.stats.completed == 2
    assert proc.stats.failed == 0
    assert proc.stats.total_tokens_in == 20
    assert proc.stats.total_tokens_out == 40


def _csv_with_rows(path, n):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input"])
        writer.writeheader()
        for i in range(n):
            writer.writerow({"input": f"row{i}"})
    return path


def test_config_limit_defaults_to_none():
    assert BatchConfig().limit is None


@pytest.mark.asyncio
async def test_process_file_limit_processes_only_first_n_rows(tmp_path):
    src = _csv_with_rows(tmp_path / "data.csv", 5)
    proc = BatchProcessor(BatchConfig(model="gpt-4o-mini", limit=2))
    proc._client = AsyncMock()
    proc._client.chat.completions.create = AsyncMock(return_value=_mock_response("ok"))

    out = tmp_path / "out.csv"
    results = await proc.process_file(src, output_path=out)

    # only the first 2 of 5 rows ran — a cheap smoke test before the full job
    assert [r.input_text for r in results] == ["row0", "row1"]
    with open(out, encoding="utf-8", newline="") as f:
        assert len(list(csv.DictReader(f))) == 2


@pytest.mark.asyncio
async def test_process_file_without_limit_processes_all_rows(tmp_path):
    src = _csv_with_rows(tmp_path / "data.csv", 3)
    proc = BatchProcessor(BatchConfig(model="gpt-4o-mini"))
    proc._client = AsyncMock()
    proc._client.chat.completions.create = AsyncMock(return_value=_mock_response("ok"))

    results = await proc.process_file(src, output_path=tmp_path / "out.csv")
    assert len(results) == 3


@pytest.mark.asyncio
async def test_process_file_limit_above_row_count_processes_all(tmp_path):
    src = _csv_with_rows(tmp_path / "data.csv", 2)
    proc = BatchProcessor(BatchConfig(model="gpt-4o-mini", limit=10))
    proc._client = AsyncMock()
    proc._client.chat.completions.create = AsyncMock(return_value=_mock_response("ok"))

    results = await proc.process_file(src, output_path=tmp_path / "out.csv")
    assert len(results) == 2


def test_config_max_cost_defaults_to_none():
    assert BatchConfig().max_cost is None


@pytest.mark.asyncio
async def test_max_cost_stops_run_early_then_checkpoint_resumes(tmp_path):
    # gpt-4o-mini input is $0.15 / 1M tokens, so each mocked row (1M prompt
    # tokens) costs $0.15; a $0.20 ceiling is crossed after the second row.
    src = _csv_with_rows(tmp_path / "data.csv", 5)
    ckpt = tmp_path / "data.ckpt"
    pricey = _mock_response("ok", prompt_tokens=1_000_000, completion_tokens=0)

    proc = BatchProcessor(BatchConfig(model="gpt-4o-mini", max_concurrent=1, max_cost=0.20))
    proc._client = AsyncMock()
    proc._client.chat.completions.create = AsyncMock(return_value=pricey)
    first = await proc.process_file(src, output_path=tmp_path / "out1.csv", checkpoint_path=ckpt)

    assert proc.stats.stopped_early is True
    done = [r for r in first if r.error is None]
    assert 1 <= len(done) < 5  # stopped before the whole file ran

    # resuming without a ceiling finishes the untouched rows from the checkpoint
    proc2 = BatchProcessor(BatchConfig(model="gpt-4o-mini", max_concurrent=1))
    proc2._client = AsyncMock()
    proc2._client.chat.completions.create = AsyncMock(return_value=_mock_response("ok"))
    second = await proc2.process_file(src, output_path=tmp_path / "out2.csv", checkpoint_path=ckpt)

    assert proc2.stats.stopped_early is False
    assert len(second) == 5
    assert all(r.error is None for r in second)


@pytest.mark.asyncio
async def test_max_cost_not_reached_processes_all(tmp_path):
    src = _csv_with_rows(tmp_path / "data.csv", 3)
    proc = BatchProcessor(BatchConfig(model="gpt-4o-mini", max_cost=1000.0))
    proc._client = AsyncMock()
    proc._client.chat.completions.create = AsyncMock(return_value=_mock_response("ok"))

    results = await proc.process_file(src, output_path=tmp_path / "out.csv")
    assert len(results) == 3
    assert proc.stats.stopped_early is False


@pytest.mark.asyncio
async def test_process_items_restores_checkpoint_stats(config, tmp_path):
    ckpt = tmp_path / "checkpoint.jsonl"
    ckpt.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "index": 0,
                        "input": "already done",
                        "output": "cached",
                        "error": None,
                        "tokens_in": 7,
                        "tokens_out": 11,
                        "latency_ms": 123.4,
                    }
                ),
                json.dumps(
                    {
                        "index": 2,
                        "input": "already failed",
                        "output": None,
                        "error": "boom",
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "latency_ms": 50.0,
                    }
                ),
            ]
        )
        + "\n"
    )

    proc = BatchProcessor(config)
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_mock_response("fresh"))
    proc._client = mock_client

    results = await proc.process_items(
        ["already done", "new item", "already failed"],
        checkpoint_path=ckpt,
    )

    assert [r.index for r in results] == [0, 1, 2]
    assert mock_client.chat.completions.create.call_count == 1
    assert proc.stats.completed == 2
    assert proc.stats.failed == 1
    assert proc.stats.total_tokens_in == 17
    assert proc.stats.total_tokens_out == 31
    assert proc.stats.total_latency_ms >= 173.4


@pytest.mark.asyncio
async def test_process_items_can_retry_only_failed_checkpoint_rows(config, tmp_path):
    ckpt = tmp_path / "checkpoint.jsonl"
    ckpt.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "index": 0,
                        "input": "already done",
                        "output": "cached",
                        "error": None,
                        "tokens_in": 7,
                        "tokens_out": 11,
                        "latency_ms": 123.4,
                    }
                ),
                json.dumps(
                    {
                        "index": 1,
                        "input": "retry me",
                        "output": None,
                        "error": "timeout",
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "latency_ms": 50.0,
                    }
                ),
            ]
        )
        + "\n"
    )

    proc = BatchProcessor(config)
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_mock_response("recovered"))
    proc._client = mock_client

    results = await proc.process_items(
        ["already done", "retry me"],
        checkpoint_path=ckpt,
        retry_failed=True,
    )

    assert mock_client.chat.completions.create.call_count == 1
    assert [result.output_text for result in results] == ["cached", "recovered"]
    assert proc.stats.completed == 2
    assert proc.stats.failed == 0

    resumed = BatchProcessor(config)
    resumed._client = mock_client
    await resumed.process_items(["already done", "retry me"], checkpoint_path=ckpt)
    assert resumed.stats.completed == 2
    assert resumed.stats.failed == 0


@pytest.mark.asyncio
async def test_failures_are_classified_and_counted(config):
    """A failing item records its error category and shows up in the breakdown."""
    proc = BatchProcessor(config)
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=TimeoutError("slow"))
    proc._client = mock_client

    results = await proc.process_items(["will time out"])

    assert results[0].error is not None
    assert results[0].error_type == "timeout"
    assert proc.stats.failed == 1
    assert proc.stats.error_breakdown == {"timeout": 1}


@pytest.mark.asyncio
async def test_error_type_persists_through_checkpoint(config, tmp_path):
    """error_type written to a checkpoint is restored into the breakdown on resume."""
    ckpt = tmp_path / "checkpoint.jsonl"
    proc = BatchProcessor(config)
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=ConnectionError("refused"))
    proc._client = mock_client

    await proc.process_items(["boom"], checkpoint_path=ckpt)
    assert proc.stats.error_breakdown == {"connection": 1}

    # Resume: the failed row is reloaded and its category rebuilt without a new call.
    resumed = BatchProcessor(config)
    resumed._client = AsyncMock()
    await resumed.process_items(["boom"], checkpoint_path=ckpt)
    assert resumed._client.chat.completions.create.call_count == 0
    assert resumed.stats.failed == 1
    assert resumed.stats.error_breakdown == {"connection": 1}


def test_restore_marks_legacy_failures_as_unknown(config, tmp_path):
    """A failed checkpoint row with no error_type counts as 'unknown' on restore."""
    ckpt = tmp_path / "checkpoint.jsonl"
    ckpt.write_text(
        json.dumps({"index": 0, "input": "x", "output": None, "error": "legacy failure"}) + "\n",
        encoding="utf-8",
    )

    proc = BatchProcessor(config)
    proc._load_checkpoint(ckpt)
    proc._restore_stats_from_results()

    assert proc.stats.failed == 1
    assert proc.stats.error_breakdown == {"unknown": 1}


@pytest.mark.asyncio
async def test_checkpoint_rejects_changed_input(config, tmp_path):
    ckpt = tmp_path / "checkpoint.jsonl"
    proc = BatchProcessor(config)
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_mock_response("done"))
    proc._client = mock_client
    await proc.process_items(["original"], checkpoint_path=ckpt)

    resumed = BatchProcessor(config)
    with pytest.raises(ValueError, match="does not match the current input"):
        await resumed.process_items(["changed"], checkpoint_path=ckpt)


@pytest.mark.asyncio
async def test_checkpoint_rejects_changed_model(config, tmp_path):
    ckpt = tmp_path / "checkpoint.jsonl"
    proc = BatchProcessor(config)
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_mock_response("done"))
    proc._client = mock_client
    await proc.process_items(["same input"], checkpoint_path=ckpt)

    resumed = BatchProcessor(BatchConfig(model="different-model"))
    with pytest.raises(ValueError, match="different inputs or model settings"):
        await resumed.process_items(["same input"], checkpoint_path=ckpt)


@pytest.mark.asyncio
async def test_process_items_does_not_reuse_previous_checkpoint_path(config, tmp_path):
    ckpt = tmp_path / "checkpoint.jsonl"
    proc = BatchProcessor(config)
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_mock_response("done"))
    proc._client = mock_client
    await proc.process_items(["first"], checkpoint_path=ckpt)
    original = ckpt.read_text(encoding="utf-8")

    await proc.process_items(["second"])

    assert ckpt.read_text(encoding="utf-8") == original


@pytest.mark.asyncio
async def test_empty_choices_is_a_failure_not_an_empty_success(config):
    # An API response with no choices must be recorded as a failure, not as a
    # completed item with empty output.
    proc = BatchProcessor(config)
    empty = MagicMock()
    empty.choices = []
    proc._client = AsyncMock()
    proc._client.chat.completions.create = AsyncMock(return_value=empty)

    results = await proc.process_items(["hi"])

    assert len(results) == 1
    assert results[0].output_text is None
    assert results[0].error is not None
    assert proc.stats.failed == 1
    assert proc.stats.completed == 0


@pytest.mark.asyncio
async def test_output_format_follows_output_path_not_input_path(tmp_path):
    # A .jsonl input with an explicit .csv output must produce CSV, not JSONL.
    src = tmp_path / "in.jsonl"
    src.write_text('{"input": "hello"}\n', encoding="utf-8")
    proc = BatchProcessor(BatchConfig(model="gpt-4o-mini"))
    proc._client = AsyncMock()
    proc._client.chat.completions.create = AsyncMock(return_value=_mock_response("hi"))
    out = tmp_path / "out.csv"

    await proc.process_file(src, output_path=out)

    first_line = out.read_text(encoding="utf-8").splitlines()[0]
    # a CSV header row, not a per-line JSON object
    assert not first_line.startswith("{")
    assert "," in first_line


class TestExpectJson:
    def test_parse_plain_json(self):
        from batchllm.processor import _parse_expected_json

        parsed, reason = _parse_expected_json('{"a": 1}')
        assert parsed == {"a": 1}
        assert reason is None

    def test_parse_fenced_json(self):
        from batchllm.processor import _parse_expected_json

        parsed, reason = _parse_expected_json('```json\n{"a": 1}\n```')
        assert parsed == {"a": 1}
        assert reason is None

    def test_parse_prose_fails(self):
        from batchllm.processor import _parse_expected_json

        parsed, reason = _parse_expected_json("sure, here is your answer")
        assert parsed is None
        assert "not valid JSON" in reason

    def test_parse_empty_fails(self):
        from batchllm.processor import _parse_expected_json

        assert _parse_expected_json("")[1] == "empty response"

    def test_missing_keys(self):
        from batchllm.processor import _missing_keys

        assert _missing_keys({"a": 1}, ["a", "b"]) == ["b"]
        assert _missing_keys([1, 2], ["a"]) == ["a"]


@pytest.mark.asyncio
async def test_expect_json_retries_invalid_then_succeeds(config):
    config.expect_json = True
    proc = BatchProcessor(config)
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[
            _mock_response("let me think about that"),
            _mock_response('{"answer": 42}'),
        ]
    )
    proc._client = mock_client

    results = await proc.process_items(["hello"])

    assert results[0].error is None
    assert results[0].output_text == '{"answer": 42}'
    assert results[0].parsed_output == {"answer": 42}
    # second attempt carried the bad completion plus the correction nudge
    second_messages = mock_client.chat.completions.create.call_args_list[1].kwargs["messages"]
    assert second_messages[-2] == {"role": "assistant", "content": "let me think about that"}
    assert "valid JSON only" in second_messages[-1]["content"]


@pytest.mark.asyncio
async def test_expect_json_persistent_failure_is_classified(config):
    config.expect_json = True
    proc = BatchProcessor(config)
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_mock_response("still prose"))
    proc._client = mock_client

    results = await proc.process_items(["hello"])

    assert results[0].error_type == "invalid_response"
    assert proc.stats.failed == 1
    assert proc.stats.error_breakdown.get("invalid_response") == 1


@pytest.mark.asyncio
async def test_expect_keys_enforced(config):
    config.expect_json = True
    config.expect_keys = ["name", "score"]
    proc = BatchProcessor(config)
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[
            _mock_response('{"name": "x"}'),
            _mock_response('{"name": "x", "score": 0.9}'),
        ]
    )
    proc._client = mock_client

    results = await proc.process_items(["hello"])

    assert results[0].error is None
    assert results[0].parsed_output == {"name": "x", "score": 0.9}
    nudge = mock_client.chat.completions.create.call_args_list[1].kwargs["messages"][-1]
    assert "missing required keys: score" in nudge["content"]


@pytest.mark.asyncio
async def test_expect_json_off_leaves_prose_alone(config):
    proc = BatchProcessor(config)
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_response("plain prose answer")
    )
    proc._client = mock_client

    results = await proc.process_items(["hello"])

    assert results[0].error is None
    assert results[0].output_text == "plain prose answer"
    assert results[0].parsed_output is None
