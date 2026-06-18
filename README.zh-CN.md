[![English](https://img.shields.io/badge/lang-English-blue)](README.md)
[![PyPI version](https://img.shields.io/pypi/v/batchllm)](https://pypi.org/project/batchllm/)
[![CI](https://github.com/he-yufeng/BatchLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/he-yufeng/BatchLLM/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

# BatchLLM

**LLM API 批量处理工具。CSV 进，处理完的 CSV 出。**

把一个文件的输入喂给 BatchLLM，它会通过任何 OpenAI 兼容 API 并发发送请求，自带自动重试、限流、断点续传和费用追踪。输出一个干净的结果文件，包含回复内容、token 数和延迟统计。

别再每次都重写那套 async 重试循环了。

## 为什么需要？

每个做过批量 LLM 处理的数据科学家都写过类似的东西：

- 基于信号量的异步请求队列控制并发
- 遇到限流时指数退避重试
- 断点续传，避免崩溃后重新处理一万条数据
- 跑大任务前先估算 token 和费用

BatchLLM 把这些全打包成一个 CLI 命令和 Python API。

## 功能特性

- **并发处理** — 基于 asyncio semaphore 的可配置并行度
- **自动重试** — 指数退避，可配置最大重试次数
- **安全断点续传** — JSONL checkpoint 会校验输入和模型配置，避免误续跑到另一批任务
- **失败归因** — 失败的行按原因分类（限流、鉴权、超时、坏请求……），汇总里直接告诉你该修什么
- **费用追踪** — 实时 token 计数，内置 30+ 模型定价
- **运行前预估** — 离线估算行数、token 和费用，断点续传时还会算出还剩多少没跑
- **`--limit N` 试跑** — 只处理前 N 行，跑完整任务前先验一下 prompt 对不对
- **`--max-cost` 成本上限** — 花费达到设定的美元预算就停下；配合 `--checkpoint` 之后再续跑剩下的行
- **多种输入格式** — CSV、JSONL、纯文本
- **兼容任何 OpenAI API** — OpenAI、DeepSeek、本地模型等
- **Prompt 模板** — 用 `{input}` 占位符自定义 prompt
- **Rich 进度条** — 实时进度、吞吐量和预计完成时间

## 安装

```bash
pip install batchllm
```

## 快速开始

### 命令行

```bash
# 基本用法：处理 CSV 文件
batchllm run data.csv -m gpt-4o-mini

# 带系统提示和模板
batchllm run data.csv -m gpt-4o-mini \
  -s "You are a translator" \
  -t "Translate to French: {input}"

# 更高并发 + 自定义输出路径
batchllm run data.csv -m gpt-4o-mini -c 20 -o results.csv

# 中断后断点续传
batchllm run data.csv -m gpt-4o-mini --checkpoint data.ckpt

# 只重试 checkpoint 中失败的行，保留成功结果
batchllm run data.csv -m gpt-4o-mini --checkpoint data.ckpt --retry-failed

# 已恢复的行仍会计入最终 token、延迟和费用汇总
# 如果换了输入、模型、prompt 或采样配置，复用旧 checkpoint 会直接报错

# 跑完整任务前，先用前 5 行 smoke test 验一下 prompt
batchllm run data.csv -m gpt-4o-mini -t "Summarize: {input}" --limit 5

# 花费到 $5 就停，之后再用同一个 checkpoint 续跑剩下的行
batchllm run data.csv -m gpt-4o-mini --max-cost 5 --checkpoint data.ckpt

# 花钱调用 API 前先验证输入格式和样本数量
batchllm validate data.csv --min-items 100

# 运行前估算费用
batchllm estimate data.csv -m gpt-4o

# 使用其他 OpenAI 兼容 API
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

### 文件处理

```python
import asyncio
from batchllm import BatchProcessor, BatchConfig

config = BatchConfig(
    model="gpt-4o-mini",
    prompt_template="Summarize in one sentence: {input}",
    max_concurrent=10,
    input_column="text",        # CSV 中的输入列名
    output_column="summary",    # 输出文件的列名
)

processor = BatchProcessor(config)
results = asyncio.run(
    processor.process_file(
        "articles.csv",
        output_path="summaries.csv",
        checkpoint_path="articles.ckpt",  # 启用断点续传
    )
)
```

## 输入格式

**CSV** — 从可配置的列名读取（默认：`input`）：
```csv
input,category
"This movie was great",review
"Terrible service",complaint
```

**JSONL** — 从可配置的字段读取：
```jsonl
{"input": "This movie was great", "category": "review"}
{"input": "Terrible service", "category": "complaint"}
```

如果配置的 CSV 列名或 JSONL 字段不存在，BatchLLM 会直接报错停止。这样做是故意的：批量任务不应该把坏输入静默变成几千条空 prompt。

你也可以先离线检查输入，不调用模型、不花钱：

```bash
batchllm validate data.csv --input-column input --min-items 100
```

**纯文本** — 每行一条：
```
This movie was great
Terrible service
```

## 输出格式

输出与输入格式对应，增加了结果列：

```csv
input,output,error,error_type,tokens_in,tokens_out,latency_ms
"This movie was great","Positive sentiment","","",15,3,234.5
"Terrible service","Negative sentiment","","",12,3,198.2
```

重试用尽仍失败的行，`error` 是错误信息，`error_type` 是分类（`rate_limit`、`auth`、
`timeout`、`connection`、`bad_request`、`conflict`、`server` 或 `other`），方便你直接
筛出值得重跑的那些行。

## 行处理失败时

一次运行里失败的行往往不是同一个原因。当某些行重试用尽还是没跑通，汇总会按原因分组，
让你知道是该退避重跑、换个 key，还是回去看输入：

```
        Failure Breakdown
┌───────────────────┬───────┐
│ Cause             │ Count │
├───────────────────┼───────┤
│ Rate limit (429)  │    42 │
│ Timeout           │     6 │
│ Bad request (4xx) │     1 │
└───────────────────┴───────┘
```

这些分类会随 checkpoint 一起保存，所以续跑时仍会带上上一次遗留的失败统计。

## 运行前预估

跑大任务前先 dry-run 一下，看清规模和花费，全程不碰 API：

```bash
$ batchllm estimate data.csv -m gpt-4o -s "You are concise" -t "Summarize: {input}"
```

```
                       Run Estimate
┌────────────────────┬───────────────┐
│ Metric             │ Value         │
├────────────────────┼───────────────┤
│ File               │ data.csv      │
│ Model              │ gpt-4o        │
│ Rows               │ 10,000        │
│ Est. input tokens  │ ~1,420,000    │
│ Est. output tokens │ ~1,250,000    │
│ Est. cost          │ $16.05        │
└────────────────────┴───────────────┘
```

token 用一个透明的启发式来估：大约 4 个字符算 1 个 token，再加上每条 chat 消息的固定开销，
而且是在**完整渲染后**的 prompt 上算的，所以模板和系统提示都会算进去。不下载分词器、不联网、
不调 API。输出 token 默认按输入的 1:1 估算，可以用 `--output-ratio` 调比例，或用 `--max-tokens`
封顶。

加上 `--checkpoint` 后，只统计还没跑完的行，于是你能看到续跑实际还要花多少：

```bash
$ batchllm estimate data.csv -m gpt-4o --checkpoint data.ckpt
```

```
│ Rows               │ 10,000        │
│ Already done       │ 6,200         │
│ Previously failed  │ 130           │
│ Remaining          │ 3,800         │
│ Est. cost          │ $6.09         │
```

默认情况下失败的行被视为已经处理过（和普通续跑一致）。加上 `--retry-failed` 就把它们算成还要重跑的工作量，
行为和 `run --retry-failed` 一致。

## 支持的模型（费用追踪）

内置 30+ 模型定价：

| 厂商 | 模型 |
|------|------|
| OpenAI | gpt-5, gpt-5-mini, gpt-5-nano, gpt-4o, gpt-4o-mini, o3, o3-mini |
| Anthropic | claude-opus-4-7, claude-sonnet-4-6, claude-haiku-4-5 |
| Google | gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash |
| DeepSeek | deepseek-chat, deepseek-reasoner |
| Mistral | mistral-large-latest, mistral-small-latest |

也支持通过 Python API 传入自定义定价。

## 配置项

| 选项 | CLI 参数 | 默认值 | 说明 |
|------|----------|--------|------|
| model | `-m` | gpt-4o-mini | 模型名称 |
| system_prompt | `-s` | None | 系统提示 |
| prompt_template | `-t` | `{input}` | Prompt 模板 |
| max_concurrent | `-c` | 10 | 最大并发数 |
| max_retries | `--max-retries` | 3 | 单条最大重试 |
| max_tokens | `--max-tokens` | None | 最大输出 token |
| temperature | `--temperature` | None | 采样温度 |
| api_key | `--api-key` | 环境变量 `OPENAI_API_KEY` | API 密钥 |
| base_url | `--base-url` | 环境变量 `OPENAI_BASE_URL` | API 地址 |

## 参与贡献

```bash
git clone https://github.com/he-yufeng/BatchLLM.git
cd BatchLLM
pip install -e ".[dev]"
pytest
```

## 许可证

MIT
