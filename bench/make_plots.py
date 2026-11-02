"""Final charts and tables, built only from saved results in bench/results/kaggle-t4/.

Every number on the README and website traces back to one of those JSON files
(one Kaggle T4, 3 repeats per configuration, median reported).

    python -m bench.make_plots
Writes docs/figures/final/*.png and docs/figures/final/summary.json.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "bench" / "results" / "kaggle-t4"
OUT = PROJECT_ROOT / "docs" / "figures" / "final"

# Validated categorical slots 1 and 2 (dataviz reference palette, light surface).
ENGINE, HF = "#2a78d6", "#eb6834"
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
USERS = [1, 8, 16, 32, 64]
RATES = [0.5, 1, 2, 4]


def summary(name: str) -> dict:
    files = sorted(RESULTS.glob(f"*_{name}.json"))
    if not files:
        raise FileNotFoundError(f"no result for {name} in {RESULTS}")
    return json.loads(files[-1].read_text(encoding="utf-8"))["summary"]


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED)


def line(ax, x, y, color, label) -> None:
    ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)


def save(fig, name: str) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"[saved] {OUT / name}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hf_users = [summary(f"sweep_hf_c{c}") for c in USERS]
    eng_users = [summary(f"sweep_engine_c{c}") for c in USERS]
    hf_rate = [summary(f"sweep_rate_hf_r{r:g}") for r in RATES]
    eng_rate = [summary(f"sweep_rate_engine_r{r:g}") for r in RATES]
    ablation = {
        "HF, one request at a time": summary("sweep_ablation_hf_sequential_c32"),
        "mini-infer, no batching": summary("sweep_ablation_engine_nobatch_c32"),
        "HF, static batching (bs 32)": summary("sweep_hf_c32"),
        "mini-infer (paged + continuous batching)": summary("sweep_engine_c32"),
    }

    # 1. Headline: throughput vs users, and time to first token vs users (two charts, one axis each).
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    tp_hf = [s["throughput_tok_s"] for s in hf_users]
    tp_eng = [s["throughput_tok_s"] for s in eng_users]
    line(ax1, USERS, tp_hf, HF, "Hugging Face (best static batch size)")
    line(ax1, USERS, tp_eng, ENGINE, "mini-infer")
    for c, h, e in zip(USERS, tp_hf, tp_eng):
        if c > 1:
            ax1.annotate(f"{e / h:.2f}x", (c, e), textcoords="offset points", xytext=(0, 9),
                         ha="center", fontsize=9, color=INK_2)
    ax1.set_xticks(USERS)
    ax1.set_ylim(0, max(tp_eng) * 1.18)
    style(ax1, "Output throughput", "concurrent users", "tokens / second")
    ax1.legend(frameon=False, labelcolor=INK_2, loc="upper left")

    line(ax2, USERS, [s["ttft_ms_p50"] / 1000 for s in hf_users], HF, "Hugging Face")
    line(ax2, USERS, [s["ttft_ms_p50"] / 1000 for s in eng_users], ENGINE, "mini-infer")
    ax2.set_yscale("log")
    ax2.set_xticks(USERS)
    style(ax2, "Time to first token, median (log scale)", "concurrent users", "seconds")
    ax2.legend(frameon=False, labelcolor=INK_2, loc="upper left")
    fig.suptitle("Qwen2.5-0.5B FP16 on one Tesla T4, ShareGPT requests, median of 3 runs",
                 color=MUTED, fontsize=9, x=0.01, ha="left")
    save(fig, "throughput_vs_users.png")

    # 2. TTFT vs arrival rate: p50 and p99 as two charts.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, key, title in ((axes[0], "ttft_ms_p50", "median (p50)"), (axes[1], "ttft_ms_p99", "tail (p99)")):
        line(ax, RATES, [s[key] / 1000 for s in hf_rate], HF, "Hugging Face (bs 32)")
        line(ax, RATES, [s[key] / 1000 for s in eng_rate], ENGINE, "mini-infer")
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.set_xticks(RATES, [f"{r:g}" for r in RATES])
        style(ax, f"Time to first token, {title}", "request arrival rate (requests / second)", "seconds (log)")
        ax.legend(frameon=False, labelcolor=INK_2, loc="center left")
    fig.suptitle("Poisson arrivals, 64 requests, up to 64 in flight, Tesla T4", color=MUTED, fontsize=9,
                 x=0.01, ha="left")
    save(fig, "ttft_vs_rate.png")

    # 3. Ablation at 32 users.
    fig, ax = plt.subplots(figsize=(10, 3.6))
    names = list(ablation)
    vals = [ablation[n]["throughput_tok_s"] for n in names]
    colors = [HF, ENGINE, HF, ENGINE]
    bars = ax.barh(names, vals, color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
    base = vals[0]
    for bar, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.0f} tok/s  ({v / base:.1f}x)",
                va="center", color=INK_2, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.25)
    style(ax, "What each piece contributes (32 concurrent users)", "tokens / second", "")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    save(fig, "ablation.png")

    record = {
        "gpu": "Tesla T4",
        "users": {c: {"hf_tok_s": h["throughput_tok_s"], "engine_tok_s": e["throughput_tok_s"],
                      "speedup": e["throughput_tok_s"] / h["throughput_tok_s"],
                      "hf_ttft_p50_ms": h["ttft_ms_p50"], "engine_ttft_p50_ms": e["ttft_ms_p50"],
                      "hf_ttft_p99_ms": h["ttft_ms_p99"], "engine_ttft_p99_ms": e["ttft_ms_p99"]}
                  for c, h, e in zip(USERS, hf_users, eng_users)},
        "rates": {r: {"hf_ttft_p50_ms": h["ttft_ms_p50"], "hf_ttft_p99_ms": h["ttft_ms_p99"],
                      "engine_ttft_p50_ms": e["ttft_ms_p50"], "engine_ttft_p99_ms": e["ttft_ms_p99"],
                      "hf_tok_s": h["throughput_tok_s"], "engine_tok_s": e["throughput_tok_s"]}
                  for r, h, e in zip(RATES, hf_rate, eng_rate)},
        "ablation_c32_tok_s": {n: s["throughput_tok_s"] for n, s in ablation.items()},
    }
    (OUT / "summary.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record["users"], indent=1))


if __name__ == "__main__":
    main()
