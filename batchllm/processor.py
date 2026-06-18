"""Core batch processing engine."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from openai import AsyncOpenAI

from batchllm.cost import estimate_cost
from batchllm.failures import UNKNOWN, classify_error

logger = logging.getLogger(__name__)

# Matches {placeholder} tokens in a prompt template.
_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _render_template(template: str, input_text: str, fields: dict[str, Any] | None) -> str:
    """Fill ``{input}`` and any ``{column}``/``{field}`` placeholders in one pass.

    ``{input}`` always resolves to the input text. Other tokens resolve to the
    matching CSV column / JSONL field (when present); unknown tokens are left
    verbatim so literal braces in a template don't break the run. Single-pass
    substitution means a value that happens to contain ``{...}`` is never
    re-expanded.
    """
    values: dict[str, str] = {"input": input_text}
    for key, value in (fields or {}).items():
        values.setdefault(key, "" if value is None else str(value))
    return _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), template)


@dataclass
class BatchConfig:
    """Configuration for a batch processing job."""

    model: str = "gpt-4o-mini"
    system_prompt: str | None = None
    prompt_template: str = "{input}"
    max_concurrent: int = 10
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    timeout: float = 120.0
    max_tokens: int | None = None
    temperature: float | None = None
    api_key: str | None = None
    base_url: str | None = None
    checkpoint_every: int = 50
    input_column: str = "input"
    output_column: str = "output"
    limit: int | None = None
    max_cost: float | None = None


@dataclass
class BatchResult:
    """Result of processing a single item."""

    index: int
    input_text: str
    output_text: str | None = None
    error: str | None = None
    error_type: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0


@dataclass
class BatchStats:
    """Aggregate stats for a batch job."""

    total: int = 0
    completed: int = 0
    failed: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_latency_ms: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    error_breakdown: dict[str, int] = field(default_factory=dict)
    stopped_early: bool = False

    @property
    def success_rate(self) -> float:
        return self.completed / max(self.total, 1)

    @property
    def avg_latency_ms(self) -> float:
        done = self.completed + self.failed
        return self.total_latency_ms / max(done, 1)

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time if self.start_time else 0.0

    @property
    def items_per_second(self) -> float:
        elapsed = self.elapsed_seconds
        return (self.completed + self.failed) / max(elapsed, 0.001)


class BatchProcessor:
    """Process a batch of inputs through an LLM API.

    Handles concurrent requests, retries with exponential backoff,
    rate limiting, checkpointing, and cost tracking.
    """

    def __init__(self, config: BatchConfig | None = None):
        self.config = config or BatchConfig()
        self._client: AsyncOpenAI | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self.stats = BatchStats()
        self._results: list[BatchResult] = []
        self._checkpoint_path: Path | None = None
        self._checkpoint_fingerprint: str | None = None
        self._completed_indices: set[int] = set()

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            kwargs: dict[str, Any] = {}
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            kwargs["timeout"] = self.config.timeout
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    def _build_messages(
        self, input_text: str, fields: dict[str, Any] | None = None
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        prompt = _render_template(self.config.prompt_template, input_text, fields)
        messages.append({"role": "user", "content": prompt})
        return messages

    async def _process_one(
        self, index: int, input_text: str, fields: dict[str, Any] | None = None
    ) -> BatchResult:
        """Process a single item with retries."""
        assert self._semaphore is not None
        async with self._semaphore:
            result = BatchResult(index=index, input_text=input_text)
            messages = self._build_messages(input_text, fields)

            for attempt in range(self.config.max_retries + 1):
                start = time.monotonic()
                try:
                    client = self._get_client()

                    kwargs: dict[str, Any] = {
                        "model": self.config.model,
                        "messages": messages,
                    }
                    if self.config.max_tokens is not None:
                        kwargs["max_tokens"] = self.config.max_tokens
                    if self.config.temperature is not None:
                        kwargs["temperature"] = self.config.temperature

                    response = await client.chat.completions.create(**kwargs)

                    elapsed = (time.monotonic() - start) * 1000
                    result.latency_ms = elapsed

                    choice = response.choices[0] if response.choices else None
                    result.output_text = choice.message.content if choice else None

                    usage = response.usage
                    if usage:
                        result.tokens_in = usage.prompt_tokens
                        result.tokens_out = usage.completion_tokens

                    self.stats.completed += 1
                    self.stats.total_tokens_in += result.tokens_in
                    self.stats.total_tokens_out += result.tokens_out
                    self.stats.total_latency_ms += result.latency_ms
                    return result

                except Exception as exc:
                    if attempt < self.config.max_retries:
                        delay = min(
                            self.config.retry_base_delay * (2**attempt),
                            self.config.retry_max_delay,
                        )
                        logger.warning(
                            "Item %d attempt %d failed: %s. Retrying in %.1fs",
                            index,
                            attempt + 1,
                            exc,
                            delay,
                        )
                        await asyncio.sleep(delay)
                    else:
                        result.error = str(exc)
                        result.error_type = classify_error(exc)
                        result.latency_ms = (time.monotonic() - start) * 1000
                        self.stats.failed += 1
                        self.stats.total_latency_ms += result.latency_ms
                        self.stats.error_breakdown[result.error_type] = (
                            self.stats.error_breakdown.get(result.error_type, 0) + 1
                        )
                        logger.error(
                            "Item %d failed after %d attempts: %s",
                            index,
                            attempt + 1,
                            exc,
                        )
                        return result

            # shouldn't reach here, but just in case
            return result

    def _make_checkpoint_fingerprint(self, items: list[str]) -> str:
        payload = {
            "items": items,
            "model": self.config.model,
            "system_prompt": self.config.system_prompt,
            "prompt_template": self.config.prompt_template,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "base_url": self.config.base_url,
            "max_retries": self.config.max_retries,
            "timeout": self.config.timeout,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load_checkpoint(
        self,
        checkpoint_path: Path,
        expected_fingerprint: str | None = None,
        items: list[str] | None = None,
        retry_failed: bool = False,
    ) -> None:
        """Load previously completed results from checkpoint."""
        if not checkpoint_path.exists():
            return
        try:
            loaded: dict[int, BatchResult] = {}
            stored_fingerprint: str | None = None
            with open(checkpoint_path, encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    if "_batchllm_checkpoint" in data:
                        stored_fingerprint = data.get("fingerprint")
                        continue
                    idx = data["index"]
                    if items is not None:
                        if idx >= len(items) or data.get("input", "") != items[idx]:
                            raise ValueError(
                                "Checkpoint does not match the current input. "
                                "Use a new checkpoint path for this batch."
                            )
                    loaded[idx] = BatchResult(
                        index=idx,
                        input_text=data.get("input", ""),
                        output_text=data.get("output"),
                        error=data.get("error"),
                        error_type=data.get("error_type"),
                        tokens_in=data.get("tokens_in", 0),
                        tokens_out=data.get("tokens_out", 0),
                        latency_ms=data.get("latency_ms", 0),
                    )
            if (
                expected_fingerprint
                and stored_fingerprint
                and stored_fingerprint != expected_fingerprint
            ):
                raise ValueError(
                    "Checkpoint was created for different inputs or model settings. "
                    "Use a new checkpoint path for this batch."
                )
            if expected_fingerprint and loaded and not stored_fingerprint:
                logger.warning(
                    "Loaded a legacy checkpoint after verifying its input rows; "
                    "model and prompt settings cannot be verified."
                )
            reusable = {
                index: result
                for index, result in loaded.items()
                if not (retry_failed and result.error is not None)
            }
            self._completed_indices.update(reusable)
            self._results.extend(sorted(reusable.values(), key=lambda r: r.index))
            logger.info("Loaded %d results from checkpoint", len(self._completed_indices))
        except ValueError:
            raise
        except Exception as exc:
            logger.warning("Failed to load checkpoint: %s", exc)

    def _restore_stats_from_results(self) -> None:
        """Rebuild stats for rows loaded from checkpoint."""
        self.stats.completed = len([r for r in self._results if r.error is None])
        self.stats.failed = len([r for r in self._results if r.error is not None])
        self.stats.total_tokens_in = sum(r.tokens_in for r in self._results)
        self.stats.total_tokens_out = sum(r.tokens_out for r in self._results)
        self.stats.total_latency_ms = sum(r.latency_ms for r in self._results)
        breakdown: dict[str, int] = {}
        for r in self._results:
            if r.error is not None:
                category = r.error_type or UNKNOWN
                breakdown[category] = breakdown.get(category, 0) + 1
        self.stats.error_breakdown = breakdown

    def _save_checkpoint(self, result: BatchResult) -> None:
        """Append a result to the checkpoint file."""
        if self._checkpoint_path is None:
            return
        try:
            self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._checkpoint_path, "a", encoding="utf-8") as f:
                if self._checkpoint_path.stat().st_size == 0 and self._checkpoint_fingerprint:
                    metadata = {
                        "_batchllm_checkpoint": 1,
                        "fingerprint": self._checkpoint_fingerprint,
                    }
                    f.write(json.dumps(metadata) + "\n")
                data = {
                    "index": result.index,
                    "input": result.input_text,
                    "output": result.output_text,
                    "error": result.error,
                    "error_type": result.error_type,
                    "tokens_in": result.tokens_in,
                    "tokens_out": result.tokens_out,
                    "latency_ms": result.latency_ms,
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Failed to save checkpoint: %s", exc)

    def _cost_ceiling_reached(self) -> bool:
        """True when a ``max_cost`` budget is set and the spend so far has hit it.

        Cost is estimated from the tokens billed so far. Unknown model pricing
        yields no estimate, so the ceiling cannot be enforced and the run
        proceeds (the pre-run estimate already flags unpriced models).
        """
        if self.config.max_cost is None:
            return False
        spent = estimate_cost(
            self.config.model,
            self.stats.total_tokens_in,
            self.stats.total_tokens_out,
        )
        return spent is not None and spent >= self.config.max_cost

    async def process_items(
        self,
        items: list[str],
        on_progress: Callable[[BatchResult, BatchStats], None] | None = None,
        checkpoint_path: str | Path | None = None,
        retry_failed: bool = False,
        fields: list[dict[str, Any]] | None = None,
    ) -> list[BatchResult]:
        """Process a list of input strings through the LLM.

        Args:
            items: List of input texts.
            on_progress: Callback for each completed item.
            checkpoint_path: Path for checkpoint file (enables resume).
            retry_failed: Reprocess failed rows while reusing successful rows.
            fields: Optional per-item field maps (CSV columns / JSONL fields)
                exposed to the prompt template as ``{column}`` placeholders.

        Returns:
            List of BatchResult objects.
        """
        self.stats = BatchStats(total=len(items), start_time=time.time())
        self._results = []
        self._completed_indices = set()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._checkpoint_path = None
        self._checkpoint_fingerprint = None

        if checkpoint_path:
            self._checkpoint_path = Path(checkpoint_path)
            self._checkpoint_fingerprint = self._make_checkpoint_fingerprint(items)
            self._load_checkpoint(
                self._checkpoint_path,
                expected_fingerprint=self._checkpoint_fingerprint,
                items=items,
                retry_failed=retry_failed,
            )
            self._restore_stats_from_results()

        # build task list, skipping already-done items
        tasks = []
        for i, text in enumerate(items):
            if i in self._completed_indices:
                continue
            item_fields = fields[i] if fields is not None and i < len(fields) else None
            tasks.append(asyncio.ensure_future(self._process_one(i, text, item_fields)))

        # run all tasks concurrently (semaphore controls parallelism)
        try:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                self._results.append(result)
                self._save_checkpoint(result)
                if on_progress:
                    on_progress(result, self.stats)
                if self._cost_ceiling_reached():
                    self.stats.stopped_early = True
                    break
        finally:
            # if we stopped early (or errored), cancel the still-pending tasks
            # so they don't run on after the budget is spent. Results already
            # collected are saved to the checkpoint, so a resume picks up the
            # untouched rows.
            pending = [t for t in tasks if not t.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        self.stats.end_time = time.time()

        # sort by original index
        self._results.sort(key=lambda r: r.index)
        return self._results

    async def process_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        on_progress: Callable[[BatchResult, BatchStats], None] | None = None,
        checkpoint_path: str | Path | None = None,
        retry_failed: bool = False,
    ) -> list[BatchResult]:
        """Process a CSV or JSONL file.

        Args:
            input_path: Path to input file (.csv or .jsonl).
            output_path: Path for output file. Defaults to input_path with .out suffix.
            on_progress: Progress callback.
            checkpoint_path: Checkpoint file path.
            retry_failed: Reprocess failed checkpoint rows.

        Returns:
            List of results. If ``config.limit`` is set, only the first
            ``limit`` rows of the file are processed (a cheap smoke test
            before committing to the full job).
        """
        input_path = Path(input_path)
        items, fields = self._read_input(input_path)

        if self.config.limit is not None:
            items = items[: self.config.limit]
            fields = fields[: self.config.limit]

        results = await self.process_items(
            items, on_progress, checkpoint_path, retry_failed, fields=fields
        )

        if output_path is None:
            suffix = input_path.suffix
            output_path = input_path.with_suffix(f".out{suffix}")

        self._write_output(Path(output_path), results, input_path)
        return results

    def _read_input(self, path: Path) -> tuple[list[str], list[dict[str, Any]]]:
        """Read input texts from CSV or JSONL.

        Returns the input strings and a parallel list of field maps (the full
        CSV row / JSONL object) so other columns are available to the prompt
        template as ``{column}`` placeholders. Plain-text inputs have no fields.
        """
        suffix = path.suffix.lower()
        items: list[str] = []
        fields: list[dict[str, Any]] = []

        if suffix == ".jsonl":
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if isinstance(data, str):
                        items.append(data)
                        fields.append({})
                    elif isinstance(data, dict):
                        items.append(self._read_mapping_input(data, path=path))
                        fields.append(data)
                    else:
                        items.append(str(data))
                        fields.append({})
        elif suffix == ".csv":
            with open(path, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None or self.config.input_column not in reader.fieldnames:
                    available = ", ".join(reader.fieldnames or [])
                    raise ValueError(
                        f"{path}: missing input column {self.config.input_column!r}"
                        + (f" (available: {available})" if available else "")
                    )
                for row in reader:
                    items.append(str(row[self.config.input_column]))
                    fields.append(dict(row))
        else:
            # plain text, one item per line
            with open(path, encoding="utf-8") as f:
                items = [line.rstrip("\n") for line in f if line.strip()]
            fields = [{} for _ in items]

        return items, fields

    def _read_mapping_input(self, data: dict[str, Any], path: Path) -> str:
        if self.config.input_column in data:
            return str(data[self.config.input_column])
        if self.config.input_column != "text" and "text" in data:
            return str(data["text"])
        raise ValueError(f"{path}: JSONL object is missing {self.config.input_column!r}")

    def _write_output(self, path: Path, results: list[BatchResult], input_path: Path) -> None:
        """Write results to CSV or JSONL."""
        suffix = path.suffix.lower()

        if suffix == ".jsonl" or input_path.suffix.lower() == ".jsonl":
            with open(path, "w", encoding="utf-8") as f:
                for r in results:
                    data = {
                        self.config.input_column: r.input_text,
                        self.config.output_column: r.output_text,
                        "error": r.error,
                        "error_type": r.error_type,
                        "tokens_in": r.tokens_in,
                        "tokens_out": r.tokens_out,
                        "latency_ms": round(r.latency_ms, 1),
                    }
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
        else:
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        self.config.input_column,
                        self.config.output_column,
                        "error",
                        "error_type",
                        "tokens_in",
                        "tokens_out",
                        "latency_ms",
                    ],
                )
                writer.writeheader()
                for r in results:
                    writer.writerow(
                        {
                            self.config.input_column: r.input_text,
                            self.config.output_column: r.output_text or "",
                            "error": r.error or "",
                            "error_type": r.error_type or "",
                            "tokens_in": r.tokens_in,
                            "tokens_out": r.tokens_out,
                            "latency_ms": round(r.latency_ms, 1),
                        }
                    )
