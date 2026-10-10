import time

import torch

from llm_bench.metrics import calculate_throughput, summarize


def run_benchmark(model, tokenizer, prompt: str, max_new_tokens: int = 64):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to("cuda")
        for key, value in inputs.items()
    }

    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()
    start = time.perf_counter()

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
    )

    torch.cuda.synchronize()
    end = time.perf_counter()

    peak_vram_bytes = torch.cuda.max_memory_allocated()
    peak_vram_gb = peak_vram_bytes / (1024 ** 3)

    latency = end - start

    input_tokens = inputs["input_ids"].shape[1]
    total_tokens = outputs.shape[1]
    generated_tokens = total_tokens - input_tokens

    throughput = calculate_throughput(
        tokens=generated_tokens,
        seconds=latency,
    )

    text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    return {
        "text": text,
        "latency": latency,
        "generated_tokens": generated_tokens,
        "throughput": throughput,
        "peak_vram_gb": peak_vram_gb,
    }

def run_repeated_benchmark(
    model,
    tokenizer,
    prompt: str,
    runs: int = 5,
    warmup_runs: int = 1,
    max_new_tokens: int = 64,
):
    cold_result = run_benchmark(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
    )

    # Warm-up runs: 不計入正式統計
    for _ in range(warmup_runs):
        run_benchmark(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
        )

    results = []

    for _ in range(runs):
        result = run_benchmark(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
        )
        results.append(result)

    latencies = [result["latency"] for result in results]
    throughputs = [result["throughput"] for result in results]
    peak_vrams = [result["peak_vram_gb"] for result in results]

    return {
        "runs": results,
        "latency_stats": summarize(latencies),
        "throughput_stats": summarize(throughputs),
        "peak_vram_gb": max(peak_vrams),
        "generated_tokens": results[0]["generated_tokens"],
        "text": results[-1]["text"],
        "cold_latency": cold_result["latency"],
    }