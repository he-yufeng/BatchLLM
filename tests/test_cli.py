import csv

from click.testing import CliRunner

from batchllm.cli import main


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
