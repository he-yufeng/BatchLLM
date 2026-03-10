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


class TestReadInput:
    def test_read_csv(self, processor, tmp_path):
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["input", "label"])
            writer.writeheader()
            writer.writerow({"input": "hello", "label": "greeting"})
            writer.writerow({"input": "bye", "label": "farewell"})

        items = processor._read_input(csv_file)
        assert items == ["hello", "bye"]

    def test_read_jsonl(self, processor, tmp_path):
        jsonl_file = tmp_path / "data.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(json.dumps({"input": "hello"}) + "\n")
            f.write(json.dumps({"input": "world"}) + "\n")

        items = processor._read_input(jsonl_file)
        assert items == ["hello", "world"]

    def test_read_plain_text(self, processor, tmp_path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("line one\nline two\n\nline three\n")

        items = processor._read_input(txt_file)
        assert items == ["line one", "line two", "line three"]

    def test_read_jsonl_strings(self, processor, tmp_path):
        jsonl_file = tmp_path / "data.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(json.dumps("hello") + "\n")
            f.write(json.dumps("world") + "\n")

        items = processor._read_input(jsonl_file)
        assert items == ["hello", "world"]


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
        processor._write_output(out, results, tmp_path / "in.csv")

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
        processor._write_output(out, results, tmp_path / "in.jsonl")

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
