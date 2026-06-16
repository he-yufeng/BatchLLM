"""Offline estimation of rows, tokens, and cost for a batch run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from batchllm.cost import estimate_cost
from batchllm.processor import BatchConfig, BatchProcessor

# Transparent heuristic. English text runs ~4 characters per token under the
# cl100k_base/o200k_base vocabularies, and the chat format spends a few extra
# tokens per message on the role/delimiters plus a few priming the reply. These
# match the overheads in OpenAI's token-counting cookbook and keep the estimate
# close to the billed prompt without pulling in a tokenizer (or its network
# download) just to size up a job.
CHARS_PER_TOKEN = 4
TOKENS_PER_MESSAGE = 4
REPLY_PRIMING_TOKENS = 3


@dataclass
class BatchEstimate:
    """Projected work and cost for a batch run, computed without any API calls."""

    total_rows: int
    rows_to_process: int
    input_tokens: int
    output_tokens: int
    cost: float | None
    completed_rows: int = 0
    failed_rows: int = 0
    has_checkpoint: bool = False


def estimate_tokens(text: str) -> int:
    """Approximate the token count of a string at ~4 characters per token."""
    if not text:
        return 0
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def _row_tokens(config: BatchConfig, row: str, output_ratio: float) -> tuple[int, int]:
    content = estimate_tokens(config.prompt_template.replace("{input}", row))
    input_tokens = content + TOKENS_PER_MESSAGE + REPLY_PRIMING_TOKENS
    if config.system_prompt:
        input_tokens += estimate_tokens(config.system_prompt) + TOKENS_PER_MESSAGE

    output_tokens = round(content * output_ratio)
    if config.max_tokens is not None:
        output_tokens = min(output_tokens, config.max_tokens)
    return input_tokens, output_tokens


def _checkpoint_progress(
    config: BatchConfig, rows: list[str], checkpoint_path: Path, retry_failed: bool
) -> tuple[int, int, list[int]]:
    """Read an existing checkpoint and return (completed, failed, pending indices).

    The checkpoint parsing, fingerprint check, and input matching live in the
    processor, so we reuse them here rather than re-implement the format.
    """
    processor = BatchProcessor(config)
    fingerprint = processor._make_checkpoint_fingerprint(rows)
    # Load every recorded row (retry_failed=False) so both states can be counted,
    # then decide what is still pending based on the caller's retry intent.
    processor._load_checkpoint(
        checkpoint_path, expected_fingerprint=fingerprint, items=rows, retry_failed=False
    )

    recorded = processor._completed_indices
    failed_indices = {r.index for r in processor._results if r.error is not None}
    completed = len(recorded) - len(failed_indices)

    pending = {i for i in range(len(rows)) if i not in recorded}
    if retry_failed:
        pending |= failed_indices
    return completed, len(failed_indices), sorted(pending)


def estimate_batch(
    config: BatchConfig,
    rows: list[str],
    output_ratio: float = 1.0,
    checkpoint_path: str | Path | None = None,
    retry_failed: bool = False,
) -> BatchEstimate:
    """Estimate the work left in a batch run.

    Renders the prompt template and system prompt for each row, sizes the input
    with the char-based heuristic, and projects output tokens as a multiple of
    the rendered input (capped by ``max_tokens`` when set). If a checkpoint
    exists, only the rows that still need processing are counted, so the cost
    reflects what a resume would actually spend.
    """
    if output_ratio < 0:
        raise ValueError("output_ratio must be non-negative")

    completed = failed = 0
    has_checkpoint = False
    pending = list(range(len(rows)))

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        has_checkpoint = True
        completed, failed, pending = _checkpoint_progress(
            config, rows, Path(checkpoint_path), retry_failed
        )

    input_tokens = output_tokens = 0
    for i in pending:
        row_in, row_out = _row_tokens(config, rows[i], output_ratio)
        input_tokens += row_in
        output_tokens += row_out

    return BatchEstimate(
        total_rows=len(rows),
        rows_to_process=len(pending),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=estimate_cost(config.model, input_tokens, output_tokens),
        completed_rows=completed,
        failed_rows=failed,
        has_checkpoint=has_checkpoint,
    )
