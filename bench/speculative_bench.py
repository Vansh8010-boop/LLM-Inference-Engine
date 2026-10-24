"""Week 4 chart: speculative decoding speedup and acceptance rate vs draft length k.

Target Qwen2.5-1.5B, draft Qwen2.5-0.5B, greedy, single sequence (latency is
what speculative decoding improves). Prompts are ShareGPT first turns wrapped
in the chat template, so the models answer the way they were trained to.

Usage:
    python -m bench.speculative_bench --num-prompts 10 --max-tokens 128
    python -m bench.speculative_bench --target-quantize int8
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from engine.device import git_commit, gpu_info, setup_device
from engine.generate import SamplingParams, generate_stream, load_eos_ids
from engine.model import Transformer
from engine.speculative import SpecStats, SpeculativeDecoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "bench" / "results"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
WORKLOAD = PROJECT_ROOT / "bench" / "workloads" / "sharegpt_1000.jsonl"
TARGET = PROJECT_ROOT / "models" / "Qwen2.5-1.5B-Instruct"
DRAFT = PROJECT_ROOT / "models" / "Qwen2.5-0.5B-Instruct"


def timed(fn) -> tuple[float, list[int]]:
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t, out


def main() -> None:
    parser = argparse.ArgumentParser(description="Speculative decoding: speedup vs k")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--target-quantize", choices=["int8"], default=None)
    parser.add_argument("--gpu-fraction", type=float, default=0.85)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    device = setup_device(args.gpu_fraction)
    tokenizer = AutoTokenizer.from_pretrained(TARGET)
    target = Transformer.from_pretrained(TARGET, device, quantize=args.target_quantize)
    draft = Transformer.from_pretrained(DRAFT, device)
    eos = load_eos_ids(TARGET)

    with WORKLOAD.open(encoding="utf-8") as f:
        texts = [json.loads(line)["prompt"] for line in f]
    prompts = []
    for t in texts:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                            tokenize=True, return_dict=False)
        if len(ids) <= 512:
            prompts.append(ids)
        if len(prompts) == args.num_prompts:
            break

    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    plain = lambda p: list(generate_stream(target, p, params, eos, kv_cache="contiguous"))
    plain(prompts[0][:32])  # warm-up

    base_time, base_tokens, base_outputs = 0.0, 0, []
    for p in prompts:
        dt, out = timed(lambda: plain(p))
        base_time += dt
        base_tokens += len(out)
        base_outputs.append(out)
    base_tok_s = base_tokens / base_time
    print(f"[target alone] {base_tok_s:.1f} tok/s over {base_tokens} tokens")

    rows = []
    for k in args.ks:
        spec = SpeculativeDecoder(target, draft, eos, k=k)
        spec.generate(prompts[0][:32], SamplingParams(max_tokens=8))  # warm-up
        stats, total_time, total_tokens, same = SpecStats(), 0.0, 0, 0
        for p, ref in zip(prompts, base_outputs):
            dt, out = timed(lambda: spec.generate(p, params, stats))
            total_time += dt
            total_tokens += len(out)
            same += out == ref
        row = {"k": k, "tok_s": total_tokens / total_time, "speedup": (total_tokens / total_time) / base_tok_s,
               "acceptance_rate": stats.acceptance_rate, "tokens_per_round": stats.tokens_per_round,
               "identical_to_target": same / len(prompts)}
        rows.append(row)
        print(f"[k={k}] {row['tok_s']:.1f} tok/s ({row['speedup']:.2f}x), acceptance {row['acceptance_rate']:.1%}, "
              f"{row['tokens_per_round']:.2f} tokens/target pass, identical {same}/{len(prompts)}")

    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    record = {"name": "speculative_bench", "timestamp": datetime.now().isoformat(timespec="seconds"),
              "server": {**gpu_info(), "commit": git_commit()}, "config": vars(args),
              "target_alone_tok_s": base_tok_s, "results": rows}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{stamp}_speculative_bench.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ks = [r["k"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    bars = ax1.bar([str(k) for k in ks], [r["speedup"] for r in rows], color="#2e86ab")
    ax1.bar_label(bars, fmt="%.2fx")
    ax1.axhline(1.0, color="#d1495b", linestyle="--", label=f"1.5B alone ({base_tok_s:.1f} tok/s)")
    ax1.set(xlabel="draft length k", ylabel="speedup vs target alone", title="Latency speedup")
    ax1.set_ylim(0, max(1.2, max(r["speedup"] for r in rows) * 1.2))
    ax1.legend()
    ax2.plot(ks, [100 * r["acceptance_rate"] for r in rows], "o-", color="#edae49", label="acceptance rate (%)")
    ax2.set(xlabel="draft length k", ylabel="accepted draft tokens (%)", title="Acceptance", ylim=(0, 100))
    ax2b = ax2.twinx()
    ax2b.plot(ks, [r["tokens_per_round"] for r in rows], "s--", color="#2e86ab", label="tokens per target pass")
    ax2b.set_ylabel("tokens per target pass")
    ax2.legend(loc="lower left")
    ax2b.legend(loc="lower right")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    quant = f" (target {args.target_quantize})" if args.target_quantize else ""
    fig.suptitle(f"Speculative decoding: Qwen2.5-0.5B drafts, Qwen2.5-1.5B{quant} verifies, greedy, "
                 f"{torch.cuda.get_device_name(0)}", fontsize=10)
    fig.tight_layout()
    out = FIGURES_DIR / f"speculative{'_' + args.target_quantize if args.target_quantize else ''}.png"
    fig.savefig(out, dpi=150)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
