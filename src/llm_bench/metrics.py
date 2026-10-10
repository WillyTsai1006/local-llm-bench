def calculate_throughput(tokens: int, seconds: float) -> float:
    return tokens / seconds

import statistics

def summarize(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.pstdev(values),
    }