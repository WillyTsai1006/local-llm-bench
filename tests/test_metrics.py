from llm_bench.metrics import calculate_throughput


def test_calculate_throughput() -> None:
    result = calculate_throughput(
        tokens=100,
        seconds=2.0,
    )

    assert result == 50.0

from llm_bench.metrics import calculate_throughput, summarize


def test_calculate_throughput() -> None:
    result = calculate_throughput(
        tokens=100,
        seconds=2.0,
    )

    assert result == 50.0


def test_summarize() -> None:
    values = [1.0, 2.0, 3.0]

    result = summarize(values)

    assert result["mean"] == 2.0
    assert result["median"] == 2.0
    assert result["min"] == 1.0
    assert result["max"] == 3.0