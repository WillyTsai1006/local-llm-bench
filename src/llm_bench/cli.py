import typer

from rich.console import Console
from rich.table import Table

from llm_bench.gpu import get_system_info

from llm_bench.model import load_model

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
def run(
    model: str = typer.Option(..., help="Hugging Face model name"),
    prompt: str = typer.Option(..., help="Prompt text"),
) -> None:
    """Run a simple local LLM inference."""

    tokenizer, llm = load_model(model)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to("cuda")
        for key, value in inputs.items()
    }

    outputs = llm.generate(
        **inputs,
        max_new_tokens=64,
    )

    text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    console.print(text)


if __name__ == "__main__":
    app()