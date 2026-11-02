"""Week 1 chart: decode speed with vs without the KV cache.

Without a cache every step re-runs attention over the whole sequence, so a
step gets slower as the output grows; with the cache a step is one token.

Usage:
    python -m bench.kv_cache_speedup --prompt-len 128 --max-tokens 512
Writes bench/results/<timestamp>_kv_cache_speedup.json and docs/figures/kv_cache_speedup.png.
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from bench.make_workloads import WORDS
from engine.device import git_commit, gpu_info
from engine.generate import LLM, SamplingParams

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "bench" / "results"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"


def time_steps(llm: LLM, prompt_ids: list[int], max_tokens: int, use_cache: bool) -> list[float]:
    """Wall-clock ms for each generated token (the first includes prefill)."""
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    times = []
    torch.cuda.synchronize()
    last = time.perf_counter()
    kv_cache = "contiguous" if use_cache else "none"
    for _ in llm.stream(prompt_ids, params, kv_cache=kv_cache):  # .item() syncs every step
        now = time.perf_counter()
        times.append((now - last) * 1000)
        last = now
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description="KV cache vs no cache decode speed")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    llm = LLM()
    rng = random.Random(0)
    text = " ".join(rng.choice(WORDS) for _ in range(args.prompt_len * 2))
    prompt_ids = llm.tokenizer.encode(text)[: args.prompt_len]

    time_steps(llm, prompt_ids, 16, True)  # warm-up
    time_steps(llm, prompt_ids, 16, False)

    results = {}
    for name, use_cache in (("cached", True), ("uncached", False)):
        runs = [time_steps(llm, prompt_ids, args.max_tokens, use_cache) for _ in range(args.repeats)]
        per_step = [sorted(step)[len(step) // 2] for step in zip(*runs)]  # median per position
        total_s = sorted(sum(r) for r in runs)[len(runs) // 2] / 1000
        results[name] = {"per_step_ms": per_step, "total_s": total_s, "tok_s": args.max_tokens / total_s}
        print(f"[{name}] {results[name]['tok_s']:.1f} tok/s, {total_s:.2f}s for {args.max_tokens} tokens, "
              f"last step {per_step[-1]:.1f} ms")

    speedup = results["uncached"]["total_s"] / results["cached"]["total_s"]
    print(f"[speedup] {speedup:.2f}x end-to-end")

    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    record = {"name": "kv_cache_speedup", "timestamp": datetime.now().isoformat(timespec="seconds"),
              "server": {**gpu_info(), "commit": git_commit(), "engine": "mini-infer"},
              "config": vars(args), "speedup": speedup, "results": results}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{stamp}_kv_cache_speedup.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    steps = range(1, args.max_tokens + 1)
    for name, color in (("uncached", "#d1495b"), ("cached", "#2e86ab")):
        # Skip step 1 (prefill) so the decode trend is readable.
        ax1.plot(list(steps)[1:], results[name]["per_step_ms"][1:], label=name, color=color, linewidth=1.2)
    ax1.set(xlabel="generated token #", ylabel="ms per token",
            title=f"Per-token latency (prompt {args.prompt_len} tokens)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    names = ["uncached", "cached"]
    bars = ax2.bar(names, [results[n]["tok_s"] for n in names], color=["#d1495b", "#2e86ab"])
    ax2.bar_label(bars, fmt="%.1f tok/s")
    ax2.set(ylabel="tokens / second", title=f"Decode throughput: {speedup:.1f}x with KV cache")
    ax2.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Qwen2.5-0.5B FP16 on {torch.cuda.get_device_name(0)}", fontsize=10)
    fig.tight_layout()
    out = FIGURES_DIR / "kv_cache_speedup.png"
    fig.savefig(out, dpi=150)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
