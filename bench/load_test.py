"""Concurrent load generator for any OpenAI-compatible /v1/completions server.

Sends a frozen workload with up to --concurrency requests in flight, optionally
spacing arrivals as a Poisson process at --rate requests/sec. Records TTFT,
TPOT, end-to-end latency and total output throughput, then writes everything
(plus the server's GPU/commit info) to bench/results/<timestamp>_<name>.json.

Usage:
    python -m bench.load_test --workload bench/workloads/sharegpt_1000.jsonl \
        --concurrency 32 --num-requests 200 --name hf_static_bs8
"""

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "bench" / "results"


def load_workload(path: Path, n: int | None) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:n] if n else rows


async def send_one(client: httpx.AsyncClient, url: str, row: dict, args) -> dict:
    body = {"prompt": row["prompt"], "max_tokens": row["max_tokens"], "temperature": args.temperature,
            "stream": True, "ignore_eos": args.ignore_eos}
    result = {"prompt_tokens": row.get("prompt_tokens"), "max_tokens": row["max_tokens"], "error": None}
    chunks = 0
    usage = None
    start = time.perf_counter()
    first = None
    try:
        async with client.stream("POST", url, json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                msg = json.loads(data)
                if "error" in msg:
                    raise RuntimeError(msg["error"]["message"])
                if first is None:
                    first = time.perf_counter()
                chunks += 1
                usage = msg.get("usage") or usage
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    end = time.perf_counter()

    tokens = usage["completion_tokens"] if usage else chunks
    result.update(
        start=start, end=end, output_tokens=tokens,
        ttft_ms=(first - start) * 1000 if first else None,
        latency_ms=(end - start) * 1000,
        tpot_ms=((end - first) * 1000 / (tokens - 1)) if first and tokens > 1 else None,
    )
    return result


async def run_once(url: str, rows: list[dict], args) -> dict:
    limits = httpx.Limits(max_connections=args.concurrency + 4)
    sem = asyncio.Semaphore(args.concurrency)
    rng = random.Random(args.seed)

    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
        for row in rows[: args.warmup]:
            await send_one(client, url, row, args)

        async def worker(row: dict, delay: float) -> dict:
            await asyncio.sleep(delay)
            async with sem:
                return await send_one(client, url, row, args)

        delays, t = [], 0.0
        for _ in rows:
            delays.append(t)
            if args.rate != float("inf"):
                t += rng.expovariate(args.rate)

        wall_start = time.perf_counter()
        results = await asyncio.gather(*(worker(r, d) for r, d in zip(rows, delays)))
        wall = time.perf_counter() - wall_start

    return {"summary": summarize(results, wall), "requests": results}


def pct(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 2) if values else None


def summarize(results: list[dict], wall_s: float) -> dict:
    ok = [r for r in results if r["error"] is None]
    total_tokens = sum(r["output_tokens"] for r in ok)
    summary = {
        "num_requests": len(results),
        "num_errors": len(results) - len(ok),
        "wall_time_s": round(wall_s, 3),
        "output_tokens": total_tokens,
        "throughput_tok_s": round(total_tokens / wall_s, 2),
        "requests_per_s": round(len(ok) / wall_s, 3),
    }
    for key in ("ttft_ms", "tpot_ms", "latency_ms"):
        vals = [r[key] for r in ok if r[key] is not None]
        summary[f"{key}_p50"] = pct(vals, 50)
        summary[f"{key}_p99"] = pct(vals, 99)
        summary[f"{key}_mean"] = round(statistics.fmean(vals), 2) if vals else None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Concurrent load test for /v1/completions")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--num-requests", type=int, default=None, help="use the first N workload rows")
    parser.add_argument("--concurrency", type=int, default=8, help="max requests in flight (concurrent users)")
    parser.add_argument("--rate", type=float, default=float("inf"), help="Poisson arrival rate, requests/sec")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false",
                        help="let requests stop at EOS instead of generating exactly max_tokens")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1, help="report the median over repeated runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="run")
    args = parser.parse_args()

    rows = load_workload(args.workload, args.num_requests)
    server_info = httpx.get(f"{args.url}/info", timeout=30).json()
    url = f"{args.url}/v1/completions"

    runs = []
    for i in range(args.repeats):
        run = asyncio.run(run_once(url, rows, args))
        runs.append(run)
        s = run["summary"]
        print(f"[run {i + 1}/{args.repeats}] {s['throughput_tok_s']} tok/s | "
              f"TTFT p50 {s['ttft_ms_p50']} / p99 {s['ttft_ms_p99']} ms | "
              f"TPOT p50 {s['tpot_ms_p50']} ms | errors {s['num_errors']}")

    keys = [k for k, v in runs[0]["summary"].items() if isinstance(v, (int, float))]
    median = {k: statistics.median(r["summary"][k] for r in runs if r["summary"][k] is not None)
              for k in keys if all(r["summary"][k] is not None for r in runs)}

    record = {
        "name": args.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "server": server_info,
        # str() for Paths and rate=inf, which plain JSON (and the website's JSON.parse) can't hold.
        "client": {k: (str(v) if isinstance(v, Path) or v == float("inf") else v)
                   for k, v in vars(args).items()},
        "workload_sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest(),
        "summary": median,
        "runs": runs,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{args.name}.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
