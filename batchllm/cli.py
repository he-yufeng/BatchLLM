"""CLI for BatchLLM."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from batchllm.cost import estimate_cost, estimate_cost_by_model, format_cost
from batchllm.estimate import estimate_batch
from batchllm.failures import describe
from batchllm.processor import BatchConfig, BatchProcessor, BatchResult, BatchStats

console = Console()


def _make_progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


@click.group()
@click.version_option()
def main():
    """BatchLLM - Batch processing for LLM APIs."""
    pass


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output file path.")
@click.option("-m", "--model", default="gpt-4o-mini", help="Model name.")
@click.option("-s", "--system", type=str, help="System prompt.")
@click.option(
    "-t", "--template", default="{input}", help="Prompt template. Use {input} placeholder."
)
@click.option("-c", "--concurrent", default=10, type=int, help="Max concurrent requests.")
@click.option("--max-retries", default=3, type=int, help="Max retries per item.")
@click.option("--max-tokens", type=int, help="Max output tokens.")
@click.option("--temperature", type=float, help="Sampling temperature.")
@click.option("--api-key", envvar="OPENAI_API_KEY", help="API key (or set OPENAI_API_KEY).")
@click.option("--base-url", envvar="OPENAI_BASE_URL", help="API base URL for compatible providers.")
@click.option("--checkpoint", type=click.Path(), help="Checkpoint file for resume support.")
@click.option(
    "--retry-failed",
    is_flag=True,
    help="Retry failed checkpoint rows while reusing successful rows.",
)
@click.option("--input-column", default="input", help="Column name for input text.")
@click.option("--output-column", default="output", help="Column name for output text.")
@click.option(
    "--model-column",
    default=None,
    help="Column whose per-row value picks the model for that row "
    "(empty cells fall back to --model).",
)
@click.option(
    "-n",
    "--limit",
    type=int,
    help="Process only the first N rows (smoke-test a prompt before the full run).",
)
@click.option(
    "--max-cost",
    type=float,
    help="Stop the run once the spend reaches this many USD (needs a known model; "
    "use --checkpoint to resume the remaining rows).",
)
@click.option(
    "--expect-json",
    is_flag=True,
    help="Require every response to parse as JSON (fences tolerated); "
    "invalid replies are retried with a correction nudge.",
)
@click.option(
    "--expect-keys",
    type=str,
    default=None,
    help="Comma-separated top-level keys the JSON object must contain (implies --expect-json).",
)
@click.option(
    "--expect-schema",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to a JSON schema file each row's parsed output must satisfy "
    "(type/required/properties/items/enum subset; implies --expect-json).",
)
def run(
    input_file: str,
    output: str | None,
    model: str,
    system: str | None,
    template: str,
    concurrent: int,
    max_retries: int,
    max_tokens: int | None,
    temperature: float | None,
    api_key: str | None,
    base_url: str | None,
    model_column: str | None,
    checkpoint: str | None,
    retry_failed: bool,
    input_column: str,
    output_column: str,
    limit: int | None,
    max_cost: float | None,
    expect_json: bool,
    expect_keys: str | None,
    expect_schema: str | None,
):
    """Process an input file (CSV/JSONL/TXT) through an LLM."""
    if retry_failed and not checkpoint:
        raise click.UsageError("--retry-failed requires --checkpoint")
    if limit is not None and limit <= 0:
        raise click.UsageError("--limit must be greater than zero")
    if max_cost is not None and max_cost <= 0:
        raise click.UsageError("--max-cost must be greater than zero")

    keys = [k.strip() for k in (expect_keys or "").split(",") if k.strip()]
    if keys and not expect_json:
        expect_json = True

    schema = None
    if expect_schema is not None:
        import json as _json
        from pathlib import Path as _Path

        try:
            schema = _json.loads(_Path(expect_schema).read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError) as exc:
            raise click.UsageError(f"--expect-schema is not a readable JSON file: {exc}") from exc
        if not isinstance(schema, dict):
            raise click.UsageError("--expect-schema must be a JSON object")
        expect_json = True

    config = BatchConfig(
        model=model,
        system_prompt=system,
        prompt_template=template,
        max_concurrent=concurrent,
        max_retries=max_retries,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        input_column=input_column,
        output_column=output_column,
        model_column=model_column,
        limit=limit,
        max_cost=max_cost,
        expect_json=expect_json,
        expect_keys=keys or None,
        expect_schema=schema,
    )

    processor = BatchProcessor(config)

    progress = _make_progress()
    task_id = progress.add_task("Processing", total=0)

    def on_progress(result: BatchResult, stats: BatchStats):
        progress.update(task_id, total=stats.total, completed=stats.completed + stats.failed)

    console.print(f"[bold]Model:[/bold] {model}")
    console.print(f"[bold]Input:[/bold] {input_file}")
    console.print(f"[bold]Concurrency:[/bold] {concurrent}")
    if limit is not None:
        console.print(f"[bold]Limit:[/bold] first {limit} row(s)")
    if max_cost is not None:
        console.print(f"[bold]Cost ceiling:[/bold] {format_cost(max_cost)}")
    if checkpoint:
        console.print(f"[bold]Checkpoint:[/bold] {checkpoint}")
    console.print()

    with progress:
        asyncio.run(
            processor.process_file(
                input_file,
                output_path=output,
                on_progress=on_progress,
                checkpoint_path=checkpoint,
                retry_failed=retry_failed,
            )
        )

    stats = processor.stats
    out_path = output or str(Path(input_file).with_suffix(f".out{Path(input_file).suffix}"))
    _print_summary(stats, model, out_path)


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-m", "--model", default="gpt-4o-mini", help="Model name for cost estimation.")
@click.option("-s", "--system", type=str, help="System prompt, matching your run command.")
@click.option(
    "-t", "--template", default="{input}", help="Prompt template. Use {input} placeholder."
)
@click.option("--max-tokens", type=int, help="Cap estimated output tokens per row.")
@click.option(
    "--output-ratio",
    default=1.0,
    type=float,
    help="Estimated output tokens as a multiple of input (default 1.0).",
)
@click.option(
    "--checkpoint", type=click.Path(), help="Checkpoint file; only unfinished rows count."
)
@click.option(
    "--retry-failed",
    is_flag=True,
    help="Count failed checkpoint rows as work still to do.",
)
@click.option("--input-column", default="input", help="Column name for input text.")
def estimate(
    input_file: str,
    model: str,
    system: str | None,
    template: str,
    max_tokens: int | None,
    output_ratio: float,
    checkpoint: str | None,
    retry_failed: bool,
    input_column: str,
):
    """Estimate rows, tokens, and cost before running. Makes no API calls."""
    if output_ratio < 0:
        raise click.UsageError("--output-ratio must be non-negative")
    if retry_failed and not checkpoint:
        raise click.UsageError("--retry-failed requires --checkpoint")

    config = BatchConfig(
        model=model,
        system_prompt=system,
        prompt_template=template,
        max_tokens=max_tokens,
        input_column=input_column,
    )
    processor = BatchProcessor(config)
    try:
        rows, item_fields = processor._read_input(Path(input_file))
        est = estimate_batch(
            config,
            rows,
            output_ratio=output_ratio,
            checkpoint_path=checkpoint,
            retry_failed=retry_failed,
            fields=item_fields,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    table = Table(title="Run Estimate")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value")
    table.add_row("File", input_file)
    table.add_row("Model", model)
    table.add_row("Rows", f"{est.total_rows:,}")
    if est.has_checkpoint:
        table.add_row("Already done", f"[green]{est.completed_rows:,}[/green]")
        table.add_row("Previously failed", f"{est.failed_rows:,}")
        table.add_row("Remaining", f"{est.rows_to_process:,}")
    table.add_row("Est. input tokens", f"~{est.input_tokens:,}")
    table.add_row("Est. output tokens", f"~{est.output_tokens:,}")
    table.add_row("Est. cost", format_cost(est.cost))

    console.print(table)
    if est.has_checkpoint and est.rows_to_process == 0:
        console.print("[green]Nothing left to run: every row is already in the checkpoint.[/green]")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--input-column", default="input", help="Column name or JSONL field for input text.")
@click.option("--min-items", default=1, type=int, help="Minimum number of rows/items required.")
def validate(input_file: str, input_column: str, min_items: int):
    """Validate an input file without making API calls."""
    if min_items <= 0:
        raise click.UsageError("--min-items must be greater than zero")

    config = BatchConfig(input_column=input_column)
    processor = BatchProcessor(config)
    try:
        items, _ = processor._read_input(Path(input_file))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    if len(items) < min_items:
        raise click.ClickException(
            f"Only found {len(items)} item(s), but --min-items is {min_items}."
        )

    preview = items[0][:120].replace("\n", " ") if items else ""
    table = Table(title="Input Validation")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value")
    table.add_row("File", input_file)
    table.add_row("Input column", input_column)
    table.add_row("Items", f"{len(items):,}")
    if preview:
        table.add_row("First item", preview)

    console.print("[green]BatchLLM input looks valid.[/green]")
    console.print(table)


def _print_summary(stats: BatchStats, model: str, output_path: str):
    if stats.tokens_by_model:
        cost = estimate_cost_by_model(stats.tokens_by_model)
    else:
        cost = estimate_cost(model, stats.total_tokens_in, stats.total_tokens_out)

    table = Table(title="Batch Complete")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value")
    table.add_row("Total Items", str(stats.total))
    table.add_row("Completed", f"[green]{stats.completed}[/green]")
    table.add_row("Failed", f"[red]{stats.failed}[/red]" if stats.failed else "0")
    table.add_row("Success Rate", f"{stats.success_rate:.1%}")
    table.add_row("Input Tokens", f"{stats.total_tokens_in:,}")
    table.add_row("Output Tokens", f"{stats.total_tokens_out:,}")
    table.add_row("Elapsed", f"{stats.elapsed_seconds:.1f}s")
    table.add_row("Throughput", f"{stats.items_per_second:.1f} items/s")
    table.add_row("Avg Latency", f"{stats.avg_latency_ms:.0f}ms")
    table.add_row("Cost", format_cost(cost))
    if stats.stopped_early:
        table.add_row("Stopped early", "[yellow]cost ceiling reached[/yellow]")
    table.add_row("Output", output_path)

    console.print()
    console.print(table)
    if stats.stopped_early:
        console.print(
            "[yellow]Run stopped at the cost ceiling; "
            "rerun with --checkpoint to finish the remaining rows.[/yellow]"
        )
    _print_failure_breakdown(stats)


def _print_failure_breakdown(stats: BatchStats):
    if not stats.error_breakdown:
        return

    table = Table(title="Failure Breakdown")
    table.add_column("Cause", style="bold red")
    table.add_column("Count", justify="right")
    for category, count in sorted(stats.error_breakdown.items(), key=lambda kv: -kv[1]):
        table.add_row(describe(category), str(count))

    console.print()
    console.print(table)


if __name__ == "__main__":
    main()
