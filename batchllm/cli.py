"""CLI for BatchLLM."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from batchllm.cost import estimate_cost, format_cost
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
@click.option("--input-column", default="input", help="Column name for input text.")
@click.option("--output-column", default="output", help="Column name for output text.")
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
    checkpoint: str | None,
    input_column: str,
    output_column: str,
):
    """Process an input file (CSV/JSONL/TXT) through an LLM."""
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
    )

    processor = BatchProcessor(config)

    progress = _make_progress()
    task_id = progress.add_task("Processing", total=0)

    def on_progress(result: BatchResult, stats: BatchStats):
        progress.update(task_id, total=stats.total, completed=stats.completed + stats.failed)

    console.print(f"[bold]Model:[/bold] {model}")
    console.print(f"[bold]Input:[/bold] {input_file}")
    console.print(f"[bold]Concurrency:[/bold] {concurrent}")
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
            )
        )

    stats = processor.stats
    out_path = output or str(Path(input_file).with_suffix(f".out{Path(input_file).suffix}"))
    _print_summary(stats, model, out_path)


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-m", "--model", default="gpt-4o-mini", help="Model name for cost estimation.")
@click.option("--input-column", default="input", help="Column name for input text.")
def estimate(input_file: str, model: str, input_column: str):
    """Estimate token count and cost without making any API calls."""
    try:
        import tiktoken
    except ImportError:
        console.print("[red]tiktoken required for estimation: pip install tiktoken[/red]")
        return

    config = BatchConfig(model=model, input_column=input_column)
    processor = BatchProcessor(config)
    items = processor._read_input(Path(input_file))

    # try to get the right encoder
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    total_tokens = sum(len(enc.encode(item)) for item in items)

    # rough estimate: output ~= input for translation/rewriting, ~0.5x for summarization
    est_output = total_tokens  # conservative 1:1 ratio

    cost = estimate_cost(model, total_tokens, est_output)

    table = Table(title="Cost Estimate")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value")
    table.add_row("Items", str(len(items)))
    table.add_row("Est. Input Tokens", f"{total_tokens:,}")
    table.add_row("Est. Output Tokens", f"~{est_output:,}")
    table.add_row("Model", model)
    table.add_row("Est. Cost", format_cost(cost))

    console.print(table)


def _print_summary(stats: BatchStats, model: str, output_path: str):
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
    table.add_row("Output", output_path)

    console.print()
    console.print(table)


if __name__ == "__main__":
    main()
