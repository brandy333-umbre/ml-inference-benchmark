# bench_inference.py
from __future__ import annotations

import argparse
import csv
import os
import time
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sample_texts import SAMPLE_TEXTS
from utils_stats import compute_latency_stats, stats_to_dict


def load_model(model_dir: str, device: str) -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model


def make_batch(texts: List[str], batch_size: int) -> List[str]:
    if not texts:
        raise ValueError("No texts provided.")
    out = []
    i = 0
    while len(out) < batch_size:
        out.append(texts[i % len(texts)])
        i += 1
    return out


def timed_forward_pass(model, tokenizer, batch_texts: List[str], max_length: int, device: str) -> float:
    inputs = tokenizer(
        batch_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Ensure fair timing for GPU: sync before and after
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(**inputs)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0  # ms


def run_benchmark(
    model_dir: str,
    device: str,
    batch_sizes: List[int],
    seq_lens: List[int],
    warmup: int,
    iters: int,
) -> List[Dict[str, Any]]:
    tokenizer, model = load_model(model_dir, device)

    results: List[Dict[str, Any]] = []
    base_texts = SAMPLE_TEXTS

    for seq_len in seq_lens:
        for bs in batch_sizes:
            batch = make_batch(base_texts, bs)

            # Warmup (stabilizes caching, JIT paths, CUDA clocks)
            for _ in range(warmup):
                _ = timed_forward_pass(model, tokenizer, batch, seq_len, device)

            latencies = []
            for _ in range(iters):
                ms = timed_forward_pass(model, tokenizer, batch, seq_len, device)
                latencies.append(ms)

            stats = compute_latency_stats(latencies)
            row = {
                "model_dir": model_dir,
                "device": device,
                "batch_size": bs,
                "max_length": seq_len,
                **stats_to_dict(stats),
                "throughput_samples_per_s": (bs / (stats.p50_ms / 1000.0)) if stats.p50_ms and stats.p50_ms > 0 else float("nan"),
            }
            results.append(row)

            print(
                f"[bs={bs:>3}, max_len={seq_len:>4}] "
                f"p50={row['p50_ms']:.2f}ms p95={row['p95_ms']:.2f}ms "
                f"thr~{row['throughput_samples_per_s']:.1f} samples/s"
            )

    return results


def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Benchmark HF text classifier inference latency/throughput.")
    parser.add_argument("--model_dir", type=str, default="clinical_nlp_model", help="Path to saved model folder.")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda. Default: auto")
    parser.add_argument("--batch_sizes", type=str, default="1,2,4,8,16", help="Comma-separated batch sizes")
    parser.add_argument("--seq_lens", type=str, default="64,128,256", help="Comma-separated max_length values")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations per config")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations per config")
    parser.add_argument("--out_csv", type=str, default="benchmarks/inference_benchmark.csv")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_sizes = parse_int_list(args.batch_sizes)
    seq_lens = parse_int_list(args.seq_lens)

    print(f"\nModel: {args.model_dir}")
    print(f"Device: {device}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Seq lens: {seq_lens}")
    print(f"Warmup: {args.warmup} | Iters: {args.iters}\n")

    rows = run_benchmark(
        model_dir=args.model_dir,
        device=device,
        batch_sizes=batch_sizes,
        seq_lens=seq_lens,
        warmup=args.warmup,
        iters=args.iters,
    )

    write_csv(rows, args.out_csv)
    print(f"\nSaved CSV -> {args.out_csv}\n")


if __name__ == "__main__":
    main()
