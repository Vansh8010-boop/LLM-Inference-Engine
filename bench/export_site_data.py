"""Export every number the website shows into site/data/results.json.

The site never hard-codes a measurement: it reads this file, which is built
only from saved result JSONs (final numbers from bench/results/kaggle-t4/,
the hardware-independent memory analysis and the laptop cross-check from
bench/results/).

    python -m bench.export_site_data
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "bench" / "results"
T4 = RESULTS / "kaggle-t4"
OUT = PROJECT_ROOT / "site" / "data" / "results.json"

USERS = [1, 8, 16, 32, 64]
RATES = [0.5, 1, 2, 4]
KEYS = ("throughput_tok_s", "ttft_ms_p50", "ttft_ms_p99", "tpot_ms_p50", "latency_ms_p50", "num_requests")


def load(directory: Path, name: str) -> dict:
    files = sorted(directory.glob(f"*_{name}.json"))
    if not files:
        raise FileNotFoundError(f"no result for {name} in {directory}")
    return json.loads(files[-1].read_text(encoding="utf-8"))


def pick(summary: dict) -> dict:
    return {k: round(summary[k], 2) for k in KEYS if k in summary}


def main() -> None:
    first = load(T4, "sweep_engine_c32")
    data = {
        "hardware": {"gpu": first["server"].get("gpu"), "driver": first["server"].get("driver"),
                     "torch": first["server"].get("torch"), "commit": first["server"].get("commit"),
                     "model": "Qwen2.5-0.5B-Instruct (FP16)", "workload": "ShareGPT, 1,000 frozen first-turn prompts",
                     "method": "3 warm-up requests, 3 runs per configuration, median reported"},
        "users": [{"users": c, "hf": pick(load(T4, f"sweep_hf_c{c}")["summary"]),
                   "engine": pick(load(T4, f"sweep_engine_c{c}")["summary"])} for c in USERS],
        "rates": [{"rate": r, "hf": pick(load(T4, f"sweep_rate_hf_r{r:g}")["summary"]),
                   "engine": pick(load(T4, f"sweep_rate_engine_r{r:g}")["summary"])} for r in RATES],
        "ablation": [
            {"label": "Hugging Face, one request at a time", "system": "hf",
             "tok_s": load(T4, "sweep_ablation_hf_sequential_c32")["summary"]["throughput_tok_s"]},
            {"label": "mini-infer, no batching", "system": "engine",
             "tok_s": load(T4, "sweep_ablation_engine_nobatch_c32")["summary"]["throughput_tok_s"]},
            {"label": "Hugging Face, static batching (bs 32)", "system": "hf",
             "tok_s": load(T4, "sweep_hf_c32")["summary"]["throughput_tok_s"]},
            {"label": "mini-infer, paged + continuous batching", "system": "engine",
             "tok_s": load(T4, "sweep_engine_c32")["summary"]["throughput_tok_s"]},
        ],
    }

    kv = load(T4, "kv_cache_speedup")
    data["kv_cache"] = {"speedup": round(kv["speedup"], 3), "prompt_len": kv["config"]["prompt_len"],
                        "cached_ms": [round(x, 2) for x in kv["results"]["cached"]["per_step_ms"]],
                        "uncached_ms": [round(x, 2) for x in kv["results"]["uncached"]["per_step_ms"]],
                        "cached_tok_s": round(kv["results"]["cached"]["tok_s"], 1),
                        "uncached_tok_s": round(kv["results"]["uncached"]["tok_s"], 1)}

    mem = load(RESULTS, "kv_memory_waste")
    data["memory"] = {"budget_gb": mem["config"]["kv_gb"], "bytes_per_token": mem["kv_bytes_per_token"],
                      "waste_pct": mem["waste_pct"], "max_concurrent": mem["max_concurrent"],
                      "waste_by_block_size": mem["paged_waste_by_block_size"]}

    quant = load(T4, "quant_bench_Qwen2.5-0.5B-Instruct")["results"]
    data["quant"] = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in quant]

    spec = load(T4, "speculative_bench")
    data["speculative"] = {"target_alone_tok_s": round(spec["target_alone_tok_s"], 1),
                           "runs": [{k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}
                                    for r in spec["results"]]}

    laptop_engine = load(RESULTS, "engine_c32")["summary"]
    laptop_hf = load(RESULTS, "hf_static_bs32_c32")["summary"]
    data["laptop"] = {"gpu": "RTX 3050 Laptop (6 GB)", "engine_tok_s": laptop_engine["throughput_tok_s"],
                      "hf_tok_s": laptop_hf["throughput_tok_s"],
                      "speedup": round(laptop_engine["throughput_tok_s"] / laptop_hf["throughput_tok_s"], 2)}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"[saved] {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
