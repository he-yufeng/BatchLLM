[![中文版](https://img.shields.io/badge/lang-中文-red)](README.zh-CN.md)
[![PyPI version](https://img.shields.io/pypi/v/batchllm)](https://pypi.org/project/batchllm/)
[![CI](https://github.com/he-yufeng/BatchLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/he-yufeng/BatchLLM/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

# BatchLLM

**Batch processing for LLM APIs. CSV in, processed CSV out.**

Feed BatchLLM a file of inputs, it fires them through any OpenAI-compatible API with concurrent requests, automatic retries, rate limiting, checkpointing, and cost tracking. Get a clean output file with results, token counts, and latency stats.

No more writing the same async retry loop for the hundredth time.

## Why?

Every data scientist who's done bulk LLM processing has written some version of this:

- Async request queue with semaphore-based concurrency
- Exponential backoff with jitter on rate limit errors
- Checkpoint/resume so you don't re-process 10k items after a crash
- Token counting and cost estimation before committing to a big job

BatchLLM packages all of this into a single CLI command and Python API.

## Features

- **Concurrent processing** — configurable parallelism with asyncio semaphore
- **Automatic retries** — exponential backoff, configurable max retries
- **Checkpoint/resume** — crash-safe JSONL checkpoints, pick up where you left off
- **Cost tracking** — real-time token counting with pricing for 30+ models
- **Cost estimation** — estimate tokens and cost before running (uses tiktoken)
- **Multiple input formats** — CSV, JSONL, plain text
- **Any OpenAI-compatible API** — OpenAI, Anthropic (via proxy), DeepSeek, local models, etc.
- **Prompt templates** — customize prompts with `{input}` placeholder
- **Rich progress bar** — live progress with throughput and ETA

## Installation

```bash
pip install batchllm
```

## Quick Start

### CLI

```bash
# Basic: process a CSV file
batchllm run data.csv -m gpt-4o-mini

# With system prompt and template
batchllm run data.csv -m gpt-4o-mini \
  -s "You are a translator" \
  -t "Translate to French: {input}"

# Higher concurrency with custom output path
batchllm run data.csv -m gpt-4o-mini -c 20 -o results.csv

# Resume from checkpoint after interruption
batchllm run data.csv -m gpt-4o-mini --checkpoint data.ckpt

# Estimate cost before running
batchllm estimate data.csv -m gpt-4o

# Use any OpenAI-compatible API
batchllm run data.csv -m deepseek-chat \
  --base-url https://api.deepseek.com/v1 \
  --api-key $DEEPSEEK_API_KEY
```

### Python API

```python
import asyncio
from batchllm import BatchProcessor, BatchConfig

config = BatchConfig(
    model="gpt-4o-mini",
    system_prompt="Classify the sentiment as positive, negative, or neutral.",
    max_concurrent=15,
    max_retries=3,
)

processor = BatchProcessor(config)

items = [
    "This product is amazing!",
    "Worst purchase ever.",
    "It's okay I guess.",
]

results = asyncio.run(processor.process_items(items))

for r in results:
    print(f"{r.input_text[:30]}... -> {r.output_text}")
    print(f"  tokens: {r.tokens_in}+{r.tokens_out}, latency: {r.latency_ms:.0f}ms")
```

### File Processing

```python
import asyncio
from batchllm import BatchProcessor, BatchConfig

config = BatchConfig(
    model="gpt-4o-mini",
    prompt_template="Summarize in one sentence: {input}",
    max_concurrent=10,
    input_column="text",
    output_column="summary",
)

processor = BatchProcessor(config)
results = asyncio.run(
    processor.process_file(
        "articles.csv",
        output_path="summaries.csv",
        checkpoint_path="articles.ckpt",
    )
)
```

## Input Formats

**CSV** — reads from a configurable column (default: `input`):
```csv
input,category
"This movie was great",review
"Terrible service",complaint
```

**JSONL** — reads from a configurable field:
```jsonl
{"input": "This movie was great", "category": "review"}
{"input": "Terrible service", "category": "complaint"}
```

**Plain text** — one item per line:
```
This movie was great
Terrible service
```

## Output Format

Output mirrors input format with added columns:

```csv
input,output,error,tokens_in,tokens_out,latency_ms
"This movie was great","Positive sentiment","",15,3,234.5
"Terrible service","Negative sentiment","",12,3,198.2
```

## Cost Estimation

Estimate before running to avoid surprises:

```bash
$ batchllm estimate data.csv -m gpt-4o
```

```
┌──────────────────┬───────────────┐
│ Metric           │ Value         │
├──────────────────┼───────────────┤
│ Items            │ 10,000        │
│ Est. Input Tokens│ 1,250,000     │
│ Est. Output      │ ~1,250,000    │
│ Model            │ gpt-4o        │
│ Est. Cost        │ $15.63        │
└──────────────────┴───────────────┘
```

## Supported Models (Cost Tracking)

Includes pricing for 30+ models:

| Provider | Models |
|----------|--------|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo, o1, o1-mini, o3-mini |
| Anthropic | claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5, claude-3.5-sonnet, claude-3-haiku |
| Google | gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash |
| DeepSeek | deepseek-chat, deepseek-reasoner |
| Mistral | mistral-large, mistral-small |

Custom pricing can be passed via the Python API.

## Configuration

| Option | CLI Flag | Default | Description |
|--------|----------|---------|-------------|
| model | `-m` | gpt-4o-mini | Model name |
| system_prompt | `-s` | None | System prompt |
| prompt_template | `-t` | `{input}` | Prompt template |
| max_concurrent | `-c` | 10 | Max parallel requests |
| max_retries | `--max-retries` | 3 | Retries per item |
| max_tokens | `--max-tokens` | None | Max output tokens |
| temperature | `--temperature` | None | Sampling temperature |
| api_key | `--api-key` | env `OPENAI_API_KEY` | API key |
| base_url | `--base-url` | env `OPENAI_BASE_URL` | API base URL |

## Contributing

```bash
git clone https://github.com/he-yufeng/BatchLLM.git
cd BatchLLM
pip install -e ".[dev]"
pytest
```

## License

MIT
