import time

import torch

from llm_bench.metrics import calculate_throughput, summarize

from llm_bench.gpu import NvmlMemoryMonitor


def run_benchmark(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to("cuda")
        for key, value in inputs.items()
    }

    torch.cuda.reset_peak_memory_stats()

    memory_monitor = NvmlMemoryMonitor()
    memory_monitor.start()

    generated_tokens_list = []

    with torch.inference_mode():
        # Prefill + first token
        torch.cuda.synchronize()
        start = time.perf_counter()

        outputs = model(
            **inputs,
            use_cache=True,
        )

        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        torch.cuda.synchronize()
        first_token_time = time.perf_counter()

        generated_tokens_list.append(next_token)

        past_key_values = outputs.past_key_values
        attention_mask = inputs["attention_mask"]
        current_token = next_token

        # Decode remaining tokens
        for _ in range(max_new_tokens - 1):
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    ),
                ],
                dim=1,
            )

            outputs = model(
                input_ids=current_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            next_token = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            generated_tokens_list.append(next_token)

            past_key_values = outputs.past_key_values
            current_token = next_token

        torch.cuda.synchronize()
        end = time.perf_counter()
        nvml_peak_used_gb = memory_monitor.stop()

    ttft = first_token_time - start
    total_latency = end - start
    decode_latency = end - first_token_time

    generated_tokens = len(generated_tokens_list)
    decode_tokens = max(generated_tokens - 1, 0)

    decode_throughput = (
        calculate_throughput(
            tokens=decode_tokens,
            seconds=decode_latency,
        )
        if decode_tokens > 0
        else 0.0
    )

    generated_ids = torch.cat(
        generated_tokens_list,
        dim=1,
    )

    full_output = torch.cat(
        [
            inputs["input_ids"],
            generated_ids,
        ],
        dim=1,
    )

    text = tokenizer.decode(
        full_output[0],
        skip_special_tokens=True,
    )

    peak_allocated_bytes = torch.cuda.max_memory_allocated()
    peak_reserved_bytes = torch.cuda.max_memory_reserved()

    peak_allocated_gb = peak_allocated_bytes / (1024**3)
    peak_reserved_gb = peak_reserved_bytes / (1024**3)

    return {
        "text": text,
        "latency": total_latency,
        "ttft": ttft,
        "decode_latency": decode_latency,
        "generated_tokens": generated_tokens,
        "throughput": decode_throughput,
        "peak_allocated_gb": peak_allocated_gb,
        "peak_reserved_gb": peak_reserved_gb,
        "nvml_peak_used_gb": nvml_peak_used_gb,
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

    latencies = [
        result["latency"]
        for result in results
    ]

    ttfts = [
        result["ttft"]
        for result in results
    ]

    decode_latencies = [
        result["decode_latency"]
        for result in results
    ]

    throughputs = [
        result["throughput"]
        for result in results
    ]

    peak_allocated = [
        result["peak_allocated_gb"]
        for result in results
    ]

    peak_reserved = [
        result["peak_reserved_gb"]
        for result in results
    ]

    nvml_peaks = [
        result["nvml_peak_used_gb"]
        for result in results
    ]

    return {
        "runs": results,
        "latency_stats": summarize(latencies),
        "ttft_stats": summarize(ttfts),
        "decode_latency_stats": summarize(decode_latencies),
        "throughput_stats": summarize(throughputs),
        "peak_allocated_gb": max(peak_allocated),
        "peak_reserved_gb": max(peak_reserved),
        "generated_tokens": results[0]["generated_tokens"],
        "text": results[-1]["text"],
        "first_inference_latency": cold_result["latency"],
        "nvml_peak_used_gb": max(nvml_peaks),
    }