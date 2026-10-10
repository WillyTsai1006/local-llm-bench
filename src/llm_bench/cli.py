import typer

from rich.console import Console
from rich.table import Table

from llm_bench.gpu import get_system_info

from llm_bench.model import load_model

from llm_bench.benchmark import run_repeated_benchmark

app = typer.Typer()
console = Console()

@app.callback()
def main() -> None:
    """Local LLM inference benchmarking tool."""
    pass

@app.command()
def system() -> None:
    """Display local Python, CUDA, and GPU information."""
    info = get_system_info()

    table = Table(title="System Information")

    table.add_column("Property")
    table.add_column("Value")

    table.add_row("Python", info["python_version"])
    table.add_row("CUDA Available", str(info["cuda_available"]))
    table.add_row("GPU", info["gpu_name"])
    table.add_row("VRAM", f"{info['vram_gb']:.2f} GB")

    console.print(table)

@app.command()
@app.command()
def run(
    model: str = typer.Option(..., help="Hugging Face model name"),
    prompt: str = typer.Option(..., help="Prompt text"),
    runs: int = typer.Option(5, help="Number of measured runs"),
    warmup_runs: int = typer.Option(1, help="Number of warm-up runs"),
    max_new_tokens: int = typer.Option(64, help="Maximum generated tokens"),
) -> None:
    """Run a local LLM inference."""

    tokenizer, llm = load_model(model)

    result = run_repeated_benchmark(
        llm,
        tokenizer,
        prompt,
        runs=runs,
        warmup_runs=warmup_runs,
        max_new_tokens=max_new_tokens,
    )

    console.print(result["text"])

    table = Table(title="Benchmark Results")

    table.add_column("Metric")
    table.add_column("Value")

    latency = result["latency_stats"]
    throughput = result["throughput_stats"]
    ttft = result["ttft_stats"]
    decode_latency = result["decode_latency_stats"]

    table.add_row("Runs", str(runs))
    table.add_row("Cold Latency",f"{result['cold_latency']:.3f} s",)
    table.add_row("Warm-up Runs", str(warmup_runs))
    table.add_row("Generated Tokens", str(result["generated_tokens"]))

    table.add_row("Mean Latency", f"{latency['mean']:.3f} s")
    table.add_row("Median Latency", f"{latency['median']:.3f} s")
    table.add_row("Min Latency", f"{latency['min']:.3f} s")
    table.add_row("Max Latency", f"{latency['max']:.3f} s")
    table.add_row("Latency Std Dev", f"{latency['std']:.3f} s")

    table.add_row(
        "Peak VRAM",
        f"{result['peak_vram_gb']:.2f} GB",
    )

    table.add_row(
        "Mean TTFT",
        f"{ttft['mean'] * 1000:.2f} ms",
    )

    table.add_row(
        "Mean Decode Latency",
        f"{decode_latency['mean']:.3f} s",
    )

    table.add_row(
        "Mean Decode Throughput",
        f"{throughput['mean']:.2f} tokens/s",
    )

    console.print(table)


if __name__ == "__main__":
    app()