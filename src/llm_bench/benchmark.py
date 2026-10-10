import time

import torch

from llm_bench.metrics import calculate_throughput


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