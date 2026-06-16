"""Tests for offline run estimation."""

import json

import pytest

from batchllm.estimate import (
    REPLY_PRIMING_TOKENS,
    TOKENS_PER_MESSAGE,
    estimate_batch,
    estimate_tokens,
)
from batchllm.processor import BatchConfig, BatchProcessor


def _write_checkpoint(path, config, rows, records):
    fingerprint = BatchProcessor(config)._make_checkpoint_fingerprint(rows)
    lines = [json.dumps({"_batchllm_checkpoint": 1, "fingerprint": fingerprint})]
    for rec in records:
        lines.append(json.dumps(rec))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_estimate_tokens_ceil():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2  # rounds up


def test_single_row_token_math():
    config = BatchConfig(model="gpt-4o-mini")  # template "{input}", no system prompt
    est = estimate_batch(config, ["hello world"])  # 11 chars -> 3 content tokens

    assert est.total_rows == 1
    assert est.rows_to_process == 1
    assert est.has_checkpoint is False
    assert est.input_tokens == 3 + TOKENS_PER_MESSAGE + REPLY_PRIMING_TOKENS
    assert est.output_tokens == 3  # 1.0 ratio
    assert est.cost is not None and est.cost > 0


def test_template_and_system_add_input_tokens():
    base = estimate_batch(BatchConfig(model="gpt-4o-mini"), ["hi"])
    templated = estimate_batch(
        BatchConfig(model="gpt-4o-mini", prompt_template="Translate to French: {input}"), ["hi"]
    )
    with_system = estimate_batch(
        BatchConfig(model="gpt-4o-mini", system_prompt="You are a translator"), ["hi"]
    )

    assert templated.input_tokens > base.input_tokens
    assert with_system.input_tokens > base.input_tokens


def test_output_ratio_scales_output():
    config = BatchConfig(model="gpt-4o-mini")
    zero = estimate_batch(config, ["hello world"], output_ratio=0.0)
    double = estimate_batch(config, ["hello world"], output_ratio=2.0)

    assert zero.output_tokens == 0
    assert double.output_tokens == 6  # 3 content tokens * 2


def test_max_tokens_caps_output():
    config = BatchConfig(model="gpt-4o-mini", max_tokens=2)
    est = estimate_batch(config, ["x" * 100])  # would be 25 content tokens uncapped

    assert est.output_tokens == 2


def test_negative_ratio_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        estimate_batch(BatchConfig(), ["hi"], output_ratio=-1.0)


def test_unknown_model_has_no_cost():
    est = estimate_batch(BatchConfig(model="totally-made-up"), ["hi"])
    assert est.cost is None


def test_empty_input():
    est = estimate_batch(BatchConfig(model="gpt-4o-mini"), [])
    assert est.total_rows == 0
    assert est.rows_to_process == 0
    assert est.input_tokens == 0
    assert est.output_tokens == 0


def test_checkpoint_recovery_counts_only_remaining(tmp_path):
    config = BatchConfig(model="gpt-4o-mini")
    rows = ["one", "two", "three", "four"]
    ckpt = tmp_path / "run.ckpt"
    _write_checkpoint(
        ckpt,
        config,
        rows,
        [
            {"index": 0, "input": "one", "output": "ok", "error": None},
            {"index": 2, "input": "three", "output": None, "error": "boom"},
        ],
    )

    est = estimate_batch(config, rows, checkpoint_path=ckpt)

    assert est.has_checkpoint is True
    assert est.total_rows == 4
    assert est.completed_rows == 1
    assert est.failed_rows == 1
    # default resume skips the failed row too, so only rows 1 and 3 remain
    assert est.rows_to_process == 2

    full = estimate_batch(config, rows)
    assert est.input_tokens < full.input_tokens


def test_checkpoint_retry_failed_counts_failed_as_pending(tmp_path):
    config = BatchConfig(model="gpt-4o-mini")
    rows = ["one", "two", "three", "four"]
    ckpt = tmp_path / "run.ckpt"
    _write_checkpoint(
        ckpt,
        config,
        rows,
        [
            {"index": 0, "input": "one", "output": "ok", "error": None},
            {"index": 2, "input": "three", "output": None, "error": "boom"},
        ],
    )

    est = estimate_batch(config, rows, checkpoint_path=ckpt, retry_failed=True)

    assert est.completed_rows == 1
    assert est.failed_rows == 1
    # rows 1 and 3 never ran, plus the failed row 2 gets retried
    assert est.rows_to_process == 3


def test_missing_checkpoint_treated_as_full_run(tmp_path):
    config = BatchConfig(model="gpt-4o-mini")
    est = estimate_batch(config, ["a", "b"], checkpoint_path=tmp_path / "nope.ckpt")

    assert est.has_checkpoint is False
    assert est.rows_to_process == 2


def test_checkpoint_rejects_changed_input(tmp_path):
    config = BatchConfig(model="gpt-4o-mini")
    rows = ["right"]
    ckpt = tmp_path / "run.ckpt"
    _write_checkpoint(
        ckpt, config, rows, [{"index": 0, "input": "wrong", "output": "ok", "error": None}]
    )

    with pytest.raises(ValueError, match="does not match the current input"):
        estimate_batch(config, rows, checkpoint_path=ckpt)
