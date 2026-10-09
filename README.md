# Local LLM Bench

A lightweight benchmarking tool for measuring local LLM inference performance on NVIDIA GPUs.

## Overview

Local LLM Bench is a command-line benchmarking tool for evaluating local large language model inference performance on NVIDIA GPUs.

The project focuses on reproducible measurements for:

- Inference latency
- Token throughput
- Peak GPU VRAM usage
- Repeated benchmark runs
- Cross-model comparison
- Structured result export

The initial implementation uses Hugging Face Transformers and PyTorch, with support for additional inference backends planned for future versions.

## Features

Planned for the first stable release:

- GPU and CUDA environment inspection
- Hugging Face causal language model loading
- Local CUDA inference
- Inference latency measurement
- Token throughput measurement
- Peak VRAM measurement
- Warm-up runs
- Repeated benchmark execution
- JSON result export
- CSV result export
- Cross-model benchmark comparison
- Basic automated tests

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- CUDA
- Typer
- Rich
- pynvml
- pytest

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd local-llm-bench
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project in editable mode:

```bash
pip install -e .
```

## Usage

### Inspect the local system

```bash
llm-bench system
```

Expected information includes:

- Python version
- PyTorch version
- CUDA availability
- CUDA version
- GPU model
- Total GPU VRAM

### Run a benchmark

```bash
llm-bench run \
  --model Qwen/Qwen2.5-0.5B \
  --prompt "Explain Linux virtual memory." \
  --max-new-tokens 128
```

Example output:

```text
Model: Qwen/Qwen2.5-0.5B
Device: NVIDIA GPU

Prompt tokens:      8
Generated tokens:   128

Total latency:       2.31 s
Throughput:          55.4 tokens/s
Peak VRAM:           2.18 GB
```

### Run repeated benchmarks

```bash
llm-bench run \
  --model Qwen/Qwen2.5-0.5B \
  --prompt "Explain Linux virtual memory." \
  --runs 5
```

### Export benchmark results

```bash
llm-bench run \
  --model Qwen/Qwen2.5-0.5B \
  --prompt "Explain Linux virtual memory." \
  --output results/qwen.json
```

### Compare benchmark results

```bash
llm-bench compare results/
```

Example comparison:

```text
MODEL              LATENCY      TOK/S      VRAM
Qwen-0.5B          2.31 s       55.4       2.18 GB
Qwen-1.5B          3.48 s       36.8       3.71 GB
TinyLlama-1.1B     3.02 s       42.1       3.19 GB
```

## Benchmark Methodology

GPU benchmarking requires care because CUDA operations are asynchronous.

The benchmark implementation will use explicit synchronization around timed regions:

```python
torch.cuda.synchronize()
```

Warm-up iterations are executed before formal measurements to reduce the impact of:

- CUDA initialization
- Kernel loading
- Memory allocation
- First-run initialization overhead

Repeated benchmark runs are used to produce more stable statistics than a single inference measurement.

Planned aggregate metrics include:

- Mean latency
- Median latency
- Minimum latency
- Maximum latency
- Standard deviation
- Mean token throughput

## Project Structure

```text
local-llm-bench/
├── README.md
├── ROADMAP.md
├── LICENSE
├── .gitignore
├── pyproject.toml
│
├── src/
│   └── llm_bench/
│       ├── __init__.py
│       ├── cli.py
│       ├── benchmark.py
│       ├── model.py
│       ├── metrics.py
│       ├── gpu.py
│       └── result.py
│
├── tests/
│   ├── test_metrics.py
│   └── test_result.py
│
├── examples/
│   └── prompts.txt
│
└── results/
    └── .gitkeep
```

## Project Status

Early development.

The first goal is to build a small but reproducible local LLM benchmarking workflow before adding serving benchmarks or distributed execution.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the development plan and future milestones.

## Future Directions

Potential future additions include:

- Time To First Token (TTFT)
- Prefill latency
- Decode latency
- Inter-token latency
- Prompt-length sweeps
- Output-length sweeps
- vLLM backend
- llama.cpp backend
- Concurrent request benchmarking
- P50 / P95 / P99 latency
- Ray-based distributed benchmarking
- Multi-GPU benchmarking

