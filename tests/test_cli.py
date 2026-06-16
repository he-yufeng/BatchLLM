import csv
import json

from click.testing import CliRunner

from batchllm.cli import main
from batchllm.processor import BatchConfig, BatchProcessor


def test_validate_accepts_csv_input(tmp_path):
    path = tmp_path / "data.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input"])
        writer.writeheader()
        writer.writerow({"input": "summarize this"})

    result = CliRunner().invoke(main, ["validate", str(path)])

    assert result.exit_code == 0, result.output
    assert "BatchLLM input looks valid" in result.output
    assert "1" in result.output


def test_validate_rejects_too_few_items(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("one\n", encoding="utf-8")

    result = CliRunner().invoke(main, ["validate", str(path), "--min-items", "2"])

    assert result.exit_code != 0
    assert "Only found 1 item" in result.output


def test_estimate_reports_cost(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("translate this\nand this too\n", encoding="utf-8")

    result = CliRunner().invoke(main, ["estimate", str(path), "-m", "gpt-4o-mini"])

    assert result.exit_code == 0, result.output
    assert "Run Estimate" in result.output
    assert "Est. cost" in result.output


def test_estimate_shows_checkpoint_recovery(tmp_path):
    config = BatchConfig(model="gpt-4o-mini")
    rows = ["one", "two", "three"]
    path = tmp_path / "data.txt"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    ckpt = tmp_path / "data.ckpt"
    fingerprint = BatchProcessor(config)._make_checkpoint_fingerprint(rows)
    ckpt.write_text(
        json.dumps({"_batchllm_checkpoint": 1, "fingerprint": fingerprint})
        + "\n"
        + json.dumps({"index": 0, "input": "one", "output": "ok", "error": None})
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main, ["estimate", str(path), "-m", "gpt-4o-mini", "--checkpoint", str(ckpt)]
    )

    assert result.exit_code == 0, result.output
    assert "Already done" in result.output
    assert "Remaining" in result.output


def test_estimate_rejects_negative_ratio(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("one\n", encoding="utf-8")

    result = CliRunner().invoke(main, ["estimate", str(path), "--output-ratio", "-1"])

    assert result.exit_code != 0
    assert "non-negative" in result.output


def test_estimate_retry_failed_requires_checkpoint(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("one\n", encoding="utf-8")

    result = CliRunner().invoke(main, ["estimate", str(path), "--retry-failed"])

    assert result.exit_code != 0
    assert "requires --checkpoint" in result.output
