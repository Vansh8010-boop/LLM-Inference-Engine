"""Head-to-head: HF baseline configurations vs the mini-infer engine.

Starts each server in turn on the same port, waits until it is healthy, runs
bench.load_test with identical settings, and shuts it down, so every number
comes from the same GPU, workload file and client.

Usage:
    python -m bench.run_comparison --num-requests 200 --concurrency 32
    python -m bench.run_comparison --only engine hf_static_bs32 --repeats 3
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "bench" / "results"

SERVERS = {
    "hf_sequential": ["-m", "server.baseline_server", "--max-batch-size", "1"],
    "hf_static_bs8": ["-m", "server.baseline_server", "--max-batch-size", "8"],
    "hf_static_bs32": ["-m", "server.baseline_server", "--max-batch-size", "32"],
    "engine": ["-m", "server.api"],
}


def wait_healthy(url: str, proc: subprocess.Popen, timeout: float = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}")
        try:
            if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise TimeoutError("server did not become healthy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the load test against each server configuration")
    parser.add_argument("--only", nargs="+", choices=list(SERVERS), default=["engine", "hf_static_bs8", "hf_static_bs32"])
    parser.add_argument("--workload", default="bench/workloads/sharegpt_1000.jsonl")
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--tag", default="", help="suffix for result names")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    summary = {}
    for name in args.only:
        print(f"\n=== {name} ===", flush=True)
        log = open(RESULTS_DIR / f"server_{name}.log", "w", encoding="utf-8")
        proc = subprocess.Popen([sys.executable, *SERVERS[name], "--port", str(args.port)],
                                cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_healthy(url, proc)
            run_name = f"{name}_c{args.concurrency}{args.tag}"
            subprocess.run([sys.executable, "-m", "bench.load_test", "--url", url, "--workload", args.workload,
                            "--num-requests", str(args.num_requests), "--concurrency", str(args.concurrency),
                            "--repeats", str(args.repeats), "--name", run_name],
                           cwd=PROJECT_ROOT, check=True)
            latest = max(RESULTS_DIR.glob(f"*_{run_name}.json"), key=lambda p: p.stat().st_mtime)
            summary[name] = json.loads(latest.read_text(encoding="utf-8"))["summary"]
        finally:
            proc.terminate()
            proc.wait(timeout=30)
            log.close()

    print("\n=== summary ===")
    print(f"{'config':18s} {'tok/s':>8s} {'TTFT p50':>10s} {'TTFT p99':>10s} {'TPOT p50':>10s} {'errors':>7s}")
    for name, s in summary.items():
        print(f"{name:18s} {s['throughput_tok_s']:8.1f} {s['ttft_ms_p50']:10.0f} {s['ttft_ms_p99']:10.0f} "
              f"{s['tpot_ms_p50']:10.1f} {s.get('num_errors', 0):7.0f}")
    if "engine" in summary:
        best = max((s["throughput_tok_s"], n) for n, s in summary.items() if n != "engine") if len(summary) > 1 else None
        if best:
            print(f"\nengine vs best baseline ({best[1]}): {summary['engine']['throughput_tok_s'] / best[0]:.2f}x throughput")


if __name__ == "__main__":
    main()
