"""Week 4 table: FP16 vs INT8 weights on memory, speed and quality.

For each precision, loads the model fresh and measures
    weights_gb         bytes of weights (tied embedding counted once)
    single_tok_s       one sequence decoding 128 tokens
    batched_tok_s      continuous batching, 32 ShareGPT prompts x 128 tokens
    ppl                WikiText-2 perplexity
    greedy_agreement   share of 32 prompts whose 32-token greedy output equals FP16's

Usage:
    python -m bench.quant_bench
    python -m bench.quant_bench --model models/Qwen2.5-1.5B-Instruct --ppl-tokens 100000
"""

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from bench.perplexity import perplexity, wikitext2_ids
from engine.device import git_commit, gpu_info, setup_device
from engine.generate import DEFAULT_MODEL, LLM, SamplingParams
from engine.llm_engine import LLMEngine
from engine.quant import weight_bytes

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "bench" / "results"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
WORKLOAD = PROJECT_ROOT / "bench" / "workloads" / "sharegpt_1000.jsonl"


def measure(model_dir: Path, quantize: str | None, prompts_text: list[str], ppl_tokens: int | None,
            reference: list[list[int]] | None) -> tuple[dict, list[list[int]]]:
    torch.cuda.reset_peak_memory_stats()
    llm = LLM(model_dir, device=torch.device("cuda"), kv_cache_gb=0.3, quantize=quantize)
    prompts = [llm.tokenizer.encode(t) for t in prompts_text]
    row = {"precision": quantize or "fp16", "weights_gb": weight_bytes(llm.model) / 1e9}

    decode = SamplingParams(max_tokens=128, ignore_eos=True)
    llm.generate(prompts[0], SamplingParams(max_tokens=8))  # warm-up
    times = []
    for p in prompts[:3]:
        torch.cuda.synchronize()
        t = time.perf_counter()
        llm.generate(p, decode)
        times.append(time.perf_counter() - t)
    row["single_tok_s"] = 128 / sorted(times)[1]

    engine = LLMEngine(llm.model, llm.tokenizer, llm.eos_ids, kv_cache_gb=0.8)
    engine.generate(prompts[:4], SamplingParams(max_tokens=8))  # warm-up
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = engine.generate(prompts, decode)
    row["batched_tok_s"] = sum(map(len, out)) / (time.perf_counter() - t)

    greedy = [llm.generate(p, SamplingParams(max_tokens=32)) for p in prompts]
    if reference is not None:
        row["greedy_agreement"] = sum(a == b for a, b in zip(greedy, reference)) / len(prompts)

    row["ppl"] = perplexity(llm.model, wikitext2_ids(llm.tokenizer), max_tokens=ppl_tokens)
    row["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
    print(f"[{row['precision']}] " + ", ".join(f"{k} {v:.3f}" for k, v in row.items() if isinstance(v, float)))

    del engine, llm
    gc.collect()
    torch.cuda.empty_cache()
    return row, greedy


def main() -> None:
    parser = argparse.ArgumentParser(description="FP16 vs INT8 weights")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--ppl-tokens", type=int, default=None, help="limit WikiText-2 tokens (default: all)")
    args = parser.parse_args()

    setup_device()
    with WORKLOAD.open(encoding="utf-8") as f:
        texts = [json.loads(line)["prompt"] for _, line in zip(range(32), f)]

    fp16, reference = measure(args.model, None, texts, args.ppl_tokens, None)
    int8, _ = measure(args.model, "int8", texts, args.ppl_tokens, reference)
    int8["ppl_increase_pct"] = 100 * (int8["ppl"] / fp16["ppl"] - 1)
    rows = [fp16, int8]

    name = args.model.name
    print(f"\n{'':8s} {'weights GB':>10s} {'single tok/s':>12s} {'batched tok/s':>13s} {'ppl':>8s} {'greedy =':>9s}")
    for r in rows:
        agree = f"{r['greedy_agreement']:.0%}" if "greedy_agreement" in r else "ref"
        print(f"{r['precision']:8s} {r['weights_gb']:10.2f} {r['single_tok_s']:12.1f} {r['batched_tok_s']:13.1f} "
              f"{r['ppl']:8.3f} {agree:>9s}")
    print(f"INT8 perplexity change: {int8['ppl_increase_pct']:+.2f}%")

    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    record = {"name": "quant_bench", "model": name, "timestamp": datetime.now().isoformat(timespec="seconds"),
              "server": {**gpu_info(), "commit": git_commit()}, "ppl_tokens": args.ppl_tokens, "results": rows}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{stamp}_quant_bench_{name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = [("weights_gb", "weights (GB)"), ("single_tok_s", "1 sequence (tok/s)"),
               ("batched_tok_s", "32 sequences (tok/s)"), ("ppl", "WikiText-2 perplexity")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (key, title) in zip(axes, metrics):
        bars = ax.bar(["FP16", "INT8"], [r[key] for r in rows], color=["#2e86ab", "#edae49"])
        ax.bar_label(bars, fmt="%.2f" if key in ("weights_gb", "ppl") else "%.0f")
        ax.set_title(title)
        ax.set_ylim(0, max(r[key] for r in rows) * 1.18)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{name} weight-only INT8 (per-channel) on {torch.cuda.get_device_name(0)}: "
                 f"perplexity {int8['ppl_increase_pct']:+.2f}%", fontsize=10)
    fig.tight_layout()
    out = FIGURES_DIR / f"quantization_{name}.png"
    fig.savefig(out, dpi=150)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
