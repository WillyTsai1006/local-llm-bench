# Local LLM Bench Roadmap

## Project Goal

Build a reproducible command-line benchmarking tool for measuring local LLM inference performance on NVIDIA GPUs.

The project should begin as a small, understandable system and grow gradually toward more advanced AI infrastructure benchmarking.

---

## Milestone 0 — Project Setup

Goal: establish a clean Python repository and development workflow.

- [X] Initialize Git repository
- [ ] Create Python project structure
- [ ] Add `pyproject.toml`
- [ ] Add `.gitignore`
- [ ] Add `README.md`
- [ ] Add `ROADMAP.md`
- [ ] Create `src/llm_bench/`
- [ ] Create `tests/`
- [ ] Create first Git commit
- [ ] Push repository to GitHub

Expected repository structure:

```text
local-llm-bench/
├── README.md
├── ROADMAP.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── src/
│   └── llm_bench/
├── tests/
├── examples/
└── results/
```

---

## Milestone 1 — System Information Command

Goal: verify that the local Python, CUDA, PyTorch, and NVIDIA environment is usable.

Implement:

```bash
llm-bench system
```

Display:

- [ ] Python version
- [ ] PyTorch version
- [ ] CUDA availability
- [ ] CUDA runtime version
- [ ] GPU name
- [ ] Total GPU VRAM

Example:

```text
Python: 3.x
PyTorch: 2.x
CUDA available: yes
CUDA version: 12.x

GPU:
NVIDIA GeForce RTX 5060 Laptop GPU

VRAM:
8 GB
```

Completion criteria:

- The CLI starts correctly.
- The GPU is detected correctly.
- CUDA availability is reported correctly.
- System information is formatted clearly.

---

## Milestone 2 — Basic Model Inference

Goal: run the first local LLM inference workload on CUDA.

Implement:

```bash
llm-bench run \
  --model Qwen/Qwen2.5-0.5B \
  --prompt "Explain Linux virtual memory."
```

Tasks:

- [ ] Load Hugging Face tokenizer
- [ ] Load causal language model
- [ ] Move model to CUDA
- [ ] Tokenize input prompt
- [ ] Run inference
- [ ] Decode generated output
- [ ] Count prompt tokens
- [ ] Count generated tokens

Completion criteria:

```text
model download
→ tokenizer load
→ model load
→ CUDA inference
→ generated text
```

---

## Milestone 3 — Benchmark Metrics

Goal: turn model inference into an actual benchmark.

Measure:

- [ ] Total latency
- [ ] Prompt token count
- [ ] Generated token count
- [ ] Tokens per second
- [ ] CUDA synchronization

Important implementation detail:

CUDA execution is asynchronous, so timing must use explicit synchronization:

```python
torch.cuda.synchronize()
```

Basic benchmark flow:

```text
synchronize
↓
start timer
↓
model inference
↓
synchronize
↓
stop timer
↓
calculate metrics
```

Suggested result object:

```python
BenchmarkResult(
    model_name=...,
    prompt_tokens=...,
    generated_tokens=...,
    latency_seconds=...,
    tokens_per_second=...,
    peak_vram_mb=...,
)
```

---

## Milestone 4 — GPU Memory Measurement

Goal: measure the memory cost of local inference.

Tasks:

- [ ] Reset CUDA peak memory statistics
- [ ] Measure peak allocated VRAM
- [ ] Convert memory values to MB / GB
- [ ] Include VRAM usage in benchmark output

Possible PyTorch APIs:

```python
torch.cuda.reset_peak_memory_stats()
torch.cuda.max_memory_allocated()
```

Completion criteria:

Benchmark output includes:

```text
Peak VRAM: 2.18 GB
```

---

## Milestone 5 — Warm-Up and Repeated Runs

Goal: make benchmark results more reproducible.

Add:

```bash
--runs 5
```

Benchmark flow:

```text
warm-up
↓
run 1
run 2
run 3
run 4
run 5
↓
aggregate statistics
```

Tasks:

- [ ] Add configurable warm-up runs
- [ ] Add configurable measured runs
- [ ] Calculate mean
- [ ] Calculate median
- [ ] Calculate minimum
- [ ] Calculate maximum
- [ ] Calculate standard deviation
- [ ] Calculate mean throughput

Example:

```bash
llm-bench run \
  --model Qwen/Qwen2.5-0.5B \
  --prompt "Explain Linux virtual memory." \
  --runs 5
```

---

## Milestone 6 — Result Export

Goal: make benchmark results reusable for comparison and analysis.

Support:

- [ ] JSON
- [ ] CSV

Example:

```bash
llm-bench run \
  --model Qwen/Qwen2.5-0.5B \
  --prompt "Explain Linux virtual memory." \
  --output results/qwen.json
```

Example JSON:

```json
{
  "model": "Qwen/Qwen2.5-0.5B",
  "device": "NVIDIA GPU",
  "prompt_tokens": 128,
  "output_tokens": 128,
  "runs": 5,
  "latency_mean": 2.31,
  "tokens_per_second": 55.4,
  "peak_vram_mb": 2180
}
```

---

## Milestone 7 — Model Comparison

Goal: compare benchmark results across models.

Implement:

```bash
llm-bench compare results/
```

Input:

```text
results/
├── qwen-0.5b.json
├── qwen-1.5b.json
└── tinyllama.json
```

Output:

```text
MODEL              LATENCY      TOK/S      VRAM
Qwen-0.5B          2.31 s       55.4       2.18 GB
Qwen-1.5B          3.48 s       36.8       3.71 GB
TinyLlama-1.1B     3.02 s       42.1       3.19 GB
```

Tasks:

- [ ] Load result files
- [ ] Validate benchmark schema
- [ ] Render comparison table
- [ ] Sort or filter results

---

## Milestone 8 — Tests and Documentation

Goal: make the project understandable and maintainable.

Testing:

- [ ] Unit test throughput calculations
- [ ] Unit test aggregate statistics
- [ ] Unit test result serialization
- [ ] Test invalid CLI input

Documentation:

- [ ] Installation instructions
- [ ] Usage examples
- [ ] Example benchmark output
- [ ] Benchmark methodology
- [ ] Hardware information
- [ ] Known limitations

The model inference path does not need excessive unit testing. Focus tests on deterministic logic.

Example:

```python
def test_tokens_per_second():
    result = calculate_throughput(tokens=100, seconds=2)
    assert result == 50
```

---

# v0.1.0 — First Stable Release

The first stable release should include:

- [ ] CLI
- [ ] GPU detection
- [ ] Hugging Face inference
- [ ] CUDA inference
- [ ] Latency measurement
- [ ] Token throughput measurement
- [ ] Peak VRAM measurement
- [ ] Warm-up runs
- [ ] Repeated benchmark runs
- [ ] JSON export
- [ ] CSV export
- [ ] Cross-model comparison
- [ ] Unit tests
- [ ] Documentation

This version should already be suitable for inclusion in a portfolio or resume.

---

# v0.2 — Detailed Inference Metrics

Add more meaningful LLM serving metrics.

Planned metrics:

- [ ] Time To First Token (TTFT)
- [ ] Prefill latency
- [ ] Decode latency
- [ ] Inter-token latency

Add benchmark sweeps:

```bash
llm-bench sweep \
  --model Qwen/Qwen2.5-0.5B \
  --prompt-lengths 32,128,512 \
  --output-lengths 32,128,256
```

Test matrix:

| Input Tokens | Output Tokens | TTFT | tok/s | VRAM |
|---:|---:|---:|---:|---:|
| 32 | 32 | TBD | TBD | TBD |
| 32 | 128 | TBD | TBD | TBD |
| 128 | 128 | TBD | TBD | TBD |
| 512 | 256 | TBD | TBD | TBD |

Also consider:

- [ ] CSV analysis output
- [ ] Matplotlib charts

---

# v0.3 — Multiple Inference Backends

Goal: compare inference frameworks, not only models.

Introduce a backend abstraction:

```text
BenchmarkEngine
├── TransformersEngine
├── VLLMEngine
└── LlamaCppEngine
```

Tasks:

- [ ] Define engine interface
- [ ] Refactor Transformers implementation
- [ ] Add vLLM backend
- [ ] Add llama.cpp backend
- [ ] Compare backend performance

This phase moves the project closer to AI infrastructure benchmarking.

---

# v0.4 — Serving Benchmarks

Goal: benchmark inference servers under realistic request load.

Metrics:

- [ ] Concurrent requests
- [ ] Batch size
- [ ] Requests per second
- [ ] Tokens per second
- [ ] P50 latency
- [ ] P95 latency
- [ ] P99 latency

Potential architecture:

```text
Benchmark Client
      │
      ▼
Inference Server
      │
      ├── request queue
      ├── batching
      └── model runtime
```

---

# Future Work

Possible later extensions:

- Ray-based distributed benchmark execution
- Multi-GPU benchmarks
- Docker environments
- Benchmark result visualization
- REST API
- Web dashboard
- Automated experiment matrices
- Hardware comparison
- Quantization comparison
- Batch-size scaling studies

---

# Development Workflow

Use the project itself to practice terminal-based development.

Recommended tmux session:

```text
llm-bench
├── 1: editor
├── 2: shell
├── 3: benchmark
└── 4: monitor
```

Suggested usage:

### Editor

```bash
nvim .
```

### Shell

```bash
git status
pytest
```

### Benchmark

```bash
llm-bench run ...
```

### GPU monitor

```bash
watch -n 1 nvidia-smi
```

---

# Git Workflow

Avoid implementing the entire project in a single commit.

Suggested commit history:

```text
Initial project structure
Add system information command
Implement model loading
Implement basic inference benchmark
Add CUDA latency measurement
Add GPU memory measurement
Add repeated benchmark runs
Add result serialization
Add model comparison command
Add tests
Document benchmark methodology
```

For each milestone:

```text
write code
→ run
→ debug
→ git diff
→ git add
→ git commit
→ push
```

