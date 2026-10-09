import typer

from rich.console import Console
from rich.table import Table

from llm_bench.gpu import get_system_info

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


if __name__ == "__main__":
    app()