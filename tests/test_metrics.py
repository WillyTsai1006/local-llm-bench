from llm_bench.metrics import calculate_throughput


def test_calculate_throughput() -> None:
    result = calculate_throughput(
        tokens=100,
        seconds=2.0,
    )

    assert result == 50.0