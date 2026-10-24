"""Week 2 chart: KV memory waste and concurrent capacity, contiguous vs paged.

Uses the real request lengths of the frozen ShareGPT workload
(prompt_tokens + max_tokens) and the engine's own BlockAllocator. Three ways
to hold the KV cache:
    reserve max context  every request preallocates MAX_CONTEXT slots
    pad to batch max     HF static batching: every row in a batch is padded to
                         the longest prompt + the largest max_tokens in it
    paged                ceil(len / block_size) blocks per request

Waste is measured at each request's final length, which is the *best* case
for the contiguous layouts (mid-generation they waste even more).
No GPU needed.

Usage:
    python -m bench.kv_memory_waste --kv-gb 3.0
"""

import argparse
import json
import random
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.make_workloads import MAX_TOTAL_TOKENS
from engine.kv_cache import BlockAllocator, kv_bytes_per_token
from engine.model import ModelConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "bench" / "results"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
MODEL_DIR = PROJECT_ROOT / "models" / "Qwen2.5-0.5B-Instruct"
WORKLOAD = PROJECT_ROOT / "bench" / "workloads" / "sharegpt_1000.jsonl"
MAX_CONTEXT = MAX_TOTAL_TOKENS  # longest request the workload allows

COLORS = {"reserve max context": "#d1495b", "pad to batch max": "#edae49", "paged": "#2e86ab"}


def waste_pct(used: int, reserved: int) -> float:
    return 100 * (reserved - used) / reserved


def padded_batches(requests: list[dict], batch_size: int) -> tuple[int, int]:
    """(used, reserved) tokens when requests are statically batched in arrival order."""
    used = reserved = 0
    for i in range(0, len(requests), batch_size):
        batch = requests[i : i + batch_size]
        row = max(r["prompt_tokens"] for r in batch) + max(r["max_tokens"] for r in batch)
        used += sum(r["prompt_tokens"] + r["max_tokens"] for r in batch)
        reserved += row * len(batch)
    return used, reserved


def capacity_padded(requests: list[dict], budget_tokens: int) -> int:
    """Largest static batch (taken in order) whose padded rows fit in the budget."""
    n, max_p, max_o = 0, 0, 0
    for r in requests:
        max_p, max_o = max(max_p, r["prompt_tokens"]), max(max_o, r["max_tokens"])
        if (n + 1) * (max_p + max_o) > budget_tokens:
            break
        n += 1
    return n


def capacity_paged(requests: list[dict], num_blocks: int, block_size: int) -> int:
    """How many requests the block allocator admits, each at its full length."""
    alloc = BlockAllocator(num_blocks, block_size)
    for i, r in enumerate(requests):
        total = r["prompt_tokens"] + r["max_tokens"]
        if not alloc.can_grow(i, total):
            return i
        alloc.grow(i, total)
    return len(requests)


def main() -> None:
    parser = argparse.ArgumentParser(description="KV memory waste: contiguous vs paged")
    parser.add_argument("--workload", type=Path, default=WORKLOAD)
    parser.add_argument("--kv-gb", type=float, default=3.0,
                        help="KV budget; ~3 GB is what a 4.8 GB cap leaves after weights and activations")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8, help="static batch size for the waste figure")
    parser.add_argument("--trials", type=int, default=200)
    args = parser.parse_args()

    requests = [json.loads(line) for line in args.workload.open(encoding="utf-8") if line.strip()]
    lengths = [r["prompt_tokens"] + r["max_tokens"] for r in requests]
    cfg = ModelConfig.from_pretrained(MODEL_DIR)
    per_token = kv_bytes_per_token(cfg)
    budget_tokens = int(args.kv_gb * 1e9) // per_token
    num_blocks = budget_tokens // args.block_size

    # --- waste over the whole workload ---
    used = sum(lengths)
    pad_used, pad_reserved = padded_batches(requests, args.batch_size)
    def paged_waste(block_size: int) -> float:
        alloc = BlockAllocator(sum(-(-n // block_size) for n in lengths), block_size)  # room for all
        for i, n in enumerate(lengths):
            alloc.grow(i, n)
        return alloc.stats()["waste_pct"]

    waste = {
        "reserve max context": waste_pct(used, MAX_CONTEXT * len(lengths)),
        "pad to batch max": waste_pct(pad_used, pad_reserved),
        "paged": paged_waste(args.block_size),
    }
    block_sweep = {bs: round(paged_waste(bs), 2) for bs in (1, 8, 16, 32, 64, 128)}

    # --- concurrent sequences in the same budget, median over random request mixes ---
    rng = random.Random(0)
    pad_caps, paged_caps = [], []
    for _ in range(args.trials):
        mix = requests[:]
        rng.shuffle(mix)
        pad_caps.append(capacity_padded(mix, budget_tokens))
        paged_caps.append(capacity_paged(mix, num_blocks, args.block_size))
    capacity = {
        "reserve max context": budget_tokens // MAX_CONTEXT,
        "pad to batch max": statistics.median(pad_caps),
        "paged": statistics.median(paged_caps),
    }

    print(f"KV budget {args.kv_gb} GB = {budget_tokens:,} tokens ({per_token} B/token), "
          f"block size {args.block_size}, mean request {statistics.fmean(lengths):.0f} tokens")
    for name in COLORS:
        print(f"  {name:20s} waste {waste[name]:5.1f}%   max concurrent sequences {capacity[name]:>6}")
    print(f"  paged waste by block size: {block_sweep}")
    gain = capacity["paged"] / capacity["reserve max context"]
    print(f"  paged fits {gain:.1f}x more sequences than reserve-max, "
          f"{capacity['paged'] / capacity['pad to batch max']:.1f}x more than padded static batching")

    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    record = {"name": "kv_memory_waste", "timestamp": datetime.now().isoformat(timespec="seconds"),
              "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "kv_bytes_per_token": per_token, "budget_tokens": budget_tokens, "max_context": MAX_CONTEXT,
              "waste_pct": waste, "max_concurrent": capacity, "paged_waste_by_block_size": block_sweep}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{stamp}_kv_memory_waste.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 4.2))
    ax0.hist(lengths, bins=40, color="#7a8b99")
    ax0.axvline(MAX_CONTEXT, color=COLORS["reserve max context"], linestyle="--", label=f"reserved: {MAX_CONTEXT}")
    ax0.axvline(statistics.fmean(lengths), color="black", linestyle=":", label=f"mean: {statistics.fmean(lengths):.0f}")
    ax0.set(xlabel="tokens per request (prompt + output)", ylabel="requests", title="ShareGPT request lengths")
    ax0.legend()

    names = list(COLORS)
    labels = ["reserve\nmax context", f"pad to batch\nmax (bs={args.batch_size})", f"paged\n({args.block_size}-tok blocks)"]
    bars = ax1.bar(labels, [waste[n] for n in names], color=[COLORS[n] for n in names])
    ax1.bar_label(bars, fmt="%.1f%%")
    ax1.set(ylabel="reserved but unused KV memory (%)", title="KV memory waste", ylim=(0, 105))

    bars = ax2.bar(labels, [capacity[n] for n in names], color=[COLORS[n] for n in names])
    ax2.bar_label(bars, fmt="%d")
    ax2.set(ylabel="sequences", title=f"Max concurrent sequences in {args.kv_gb:g} GB: {gain:.1f}x with paging")
    ax2.set_ylim(0, max(capacity.values()) * 1.15)
    for ax in (ax1, ax2):
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Qwen2.5-0.5B FP16 KV cache, 12 KiB/token, ShareGPT workload (1,000 requests)", fontsize=10)
    fig.tight_layout()
    out = FIGURES_DIR / "kv_memory_waste.png"
    fig.savefig(out, dpi=150)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
