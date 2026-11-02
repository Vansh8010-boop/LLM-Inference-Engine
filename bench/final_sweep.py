"""Week 5 final benchmark sweep: every configuration, same GPU, frozen workloads.

Runs, strictly one at a time:
  1. Throughput vs concurrent users (1, 8, 16, 32, 64): HF baseline vs engine
  2. TTFT vs request arrival rate (0.5, 1, 2, 4 req/s): HF baseline vs engine
  3. Ablation at 32 users: HF sequential, engine without batching
     (the full engine and HF bs=32 rows come from sweep 1)

Each load test runs 3 times (median reported) after 3 warm-up requests.

Built to survive a 6 GB laptop overnight:
  - GPU memory capped per server (--gpu-fraction, default 0.70 = ~4.5 GB)
  - waits for Windows to have free commit memory before starting each server
  - one server at a time, cooldown between runs so the GPU is not thermally
    throttled differently for different configurations
  - resumable: a run whose result file already exists is skipped, so after a
    crash or reboot just start it again
  - a failing server or run is logged and skipped, the sweep continues

Usage:
    python -m bench.final_sweep --dry-run      # print the plan and time estimate
    python -m bench.final_sweep                # run everything (~4-5 h on an RTX 3050 Laptop)
Progress: bench/results/sweep_logs/sweep.log
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "bench" / "results"
LOG_DIR = RESULTS_DIR / "sweep_logs"
WORKLOAD = "bench/workloads/sharegpt_1000.jsonl"

CONCURRENCY = [1, 8, 16, 32, 64]
# Enough requests for stable numbers, few enough that the slow HF baseline fits in a night.
REQUESTS_AT = {1: 20, 8: 64, 16: 96, 32: 160, 64: 256}
RATES = [0.5, 1.0, 2.0, 4.0]
RATE_REQUESTS = 64
ABLATION_REQUESTS = 32
HF_MAX_BATCH = 32  # best HF setting measured in week 3


@dataclass
class Run:
    name: str
    load_args: list[str]
    est_tokens: int  # rough output tokens, for the time estimate


@dataclass
class Group:
    """One server configuration and the load tests run against it."""
    key: str
    server_args: list[str]
    est_tok_s: float  # rough throughput, for the time estimate
    runs: list[Run] = field(default_factory=list)


def hf(batch: int) -> list[str]:
    return ["-m", "server.baseline_server", "--max-batch-size", str(batch)]


def build_plan(kv_cache_gb: float) -> list[Group]:
    avg_out = 280  # mean ShareGPT output tokens
    engine = ["-m", "server.api", "--kv-cache-gb", str(kv_cache_gb)]
    groups = {
        "hf_bs1": Group("hf_bs1", hf(1), 22),
        "hf_bs8": Group("hf_bs8", hf(8), 100),
        "hf_bs16": Group("hf_bs16", hf(16), 125),
        "hf_bs32": Group("hf_bs32", hf(HF_MAX_BATCH), 147),
        "engine": Group("engine", engine, 330),
        "engine_nobatch": Group("engine_nobatch", engine + ["--max-num-seqs", "1"], 25),
    }

    # 1. throughput vs concurrency: HF uses a batch size matching the load (capped at its best, 32)
    for c in CONCURRENCY:
        n = REQUESTS_AT[c]
        load = ["--concurrency", str(c), "--num-requests", str(n)]
        hf_key = {1: "hf_bs1", 8: "hf_bs8", 16: "hf_bs16"}.get(c, "hf_bs32")
        groups[hf_key].runs.append(Run(f"sweep_hf_c{c}", load, n * avg_out))
        groups["engine"].runs.append(Run(f"sweep_engine_c{c}", load, n * avg_out))

    # 2. TTFT vs arrival rate (Poisson arrivals, up to 64 in flight)
    for r in RATES:
        load = ["--rate", str(r), "--concurrency", "64", "--num-requests", str(RATE_REQUESTS)]
        groups["hf_bs32"].runs.append(Run(f"sweep_rate_hf_r{r:g}", load, RATE_REQUESTS * avg_out))
        groups["engine"].runs.append(Run(f"sweep_rate_engine_r{r:g}", load, RATE_REQUESTS * avg_out))

    # 3. ablation at 32 users
    load = ["--concurrency", "32", "--num-requests", str(ABLATION_REQUESTS)]
    groups["hf_bs1"].runs.append(Run("sweep_ablation_hf_sequential_c32", load, ABLATION_REQUESTS * avg_out))
    groups["engine_nobatch"].runs.append(Run("sweep_ablation_engine_nobatch_c32", load, ABLATION_REQUESTS * avg_out))

    return [g for g in groups.values() if g.runs]


# ---------------------------------------------------------------- helpers

def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "sweep.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def free_commit_gb() -> float:
    """Windows commit memory still available (RAM + page file); what ran out before."""
    if sys.platform != "win32":
        return float("inf")

    class MemStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    stat = MemStatus()
    stat.dwLength = ctypes.sizeof(MemStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullAvailPageFile / 1e9


def wait_for_memory(min_gb: float, timeout_s: float = 600) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        free = free_commit_gb()
        if free >= min_gb:
            return True
        log(f"  only {free:.1f} GB commit memory free (need {min_gb}), waiting...")
        time.sleep(30)
    return False


def gpu_status() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,clocks.sm", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "n/a"


def result_exists(name: str) -> bool:
    return any(RESULTS_DIR.glob(f"*_{name}.json"))


def wait_healthy(url: str, proc: subprocess.Popen, timeout: float = 300) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}")
        try:
            if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise TimeoutError("server did not become healthy")


def child_env(gpu_fraction: float) -> dict:
    env = dict(os.environ)
    env.update({"MINI_INFER_GPU_FRACTION": str(gpu_fraction), "PYTHONUNBUFFERED": "1",
                "TQDM_DISABLE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_OFFLINE": "1"})
    return env


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description="Week 5 final benchmark sweep")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and an estimate, run nothing")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu-fraction", type=float, default=0.70, help="GPU memory cap per server")
    parser.add_argument("--kv-cache-gb", type=float, default=1.5, help="engine KV pool (fits under the cap)")
    parser.add_argument("--min-free-gb", type=float, default=6.0, help="commit memory needed to start a server")
    parser.add_argument("--cooldown", type=int, default=45, help="seconds idle between runs")
    parser.add_argument("--not-before", default=None, help="HH:MM, wait until this local time before starting")
    parser.add_argument("--port", type=int, default=8020)
    args = parser.parse_args()

    plan = build_plan(args.kv_cache_gb)
    total_runs = sum(len(g.runs) for g in plan)
    est_s = sum(args.repeats * r.est_tokens / g.est_tok_s + args.cooldown for g in plan for r in g.runs)
    est_s += 60 * len(plan)  # server start-up

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        for g in plan:
            print(f"{g.key:16s} {' '.join(g.server_args)}")
            for r in g.runs:
                done = " (done, will skip)" if result_exists(r.name) else ""
                print(f"    {r.name:40s} {' '.join(r.load_args)}{done}")
        print(f"\n{total_runs} runs x {args.repeats} repeats, estimated {est_s / 3600:.1f} h")
        return

    if args.not_before:
        h, m = map(int, args.not_before.split(":"))
        start = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
        if datetime.now() < start:
            log(f"waiting until {args.not_before} to start")
            time.sleep((start - datetime.now()).total_seconds())

    if sys.platform == "win32":
        # Keep the laptop from sleeping until this process exits (no power settings are changed).
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

    log(f"sweep start: {total_runs} runs x {args.repeats} repeats, estimated {est_s / 3600:.1f} h, "
        f"GPU cap {args.gpu_fraction:.0%}, engine KV {args.kv_cache_gb} GB")
    url = f"http://127.0.0.1:{args.port}"
    env = child_env(args.gpu_fraction)
    summary = {}

    for g in plan:
        todo = [r for r in g.runs if not result_exists(r.name)]
        if not todo:
            log(f"{g.key}: all runs done, skipping")
            continue
        if not wait_for_memory(args.min_free_gb):
            log(f"{g.key}: SKIPPED, not enough free memory ({free_commit_gb():.1f} GB)")
            continue

        server_log = open(LOG_DIR / f"server_{g.key}.log", "a", encoding="utf-8")
        cmd = [sys.executable, *g.server_args, "--port", str(args.port), "--gpu-fraction", str(args.gpu_fraction)]
        log(f"{g.key}: starting server ({len(todo)} runs)")
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=server_log, stderr=subprocess.STDOUT, env=env)
        try:
            wait_healthy(url, proc)
            for r in todo:
                log(f"  {r.name}: start, GPU {gpu_status()}")
                t = time.time()
                run_log = open(LOG_DIR / f"{r.name}.log", "w", encoding="utf-8")
                result = subprocess.run(
                    [sys.executable, "-m", "bench.load_test", "--url", url, "--workload", WORKLOAD,
                     "--repeats", str(args.repeats), "--name", r.name, *r.load_args],
                    cwd=PROJECT_ROOT, stdout=run_log, stderr=subprocess.STDOUT, env=env)
                run_log.close()
                if result.returncode != 0:
                    log(f"  {r.name}: FAILED (exit {result.returncode}), see sweep_logs/{r.name}.log")
                elif proc.poll() is not None:
                    log(f"  {r.name}: server died during the run, see sweep_logs/server_{g.key}.log")
                    break
                else:
                    latest = max(RESULTS_DIR.glob(f"*_{r.name}.json"), key=lambda p: p.stat().st_mtime)
                    s = json.loads(latest.read_text(encoding="utf-8"))["summary"]
                    summary[r.name] = s
                    log(f"  {r.name}: {s['throughput_tok_s']:.1f} tok/s, TTFT p50 {s['ttft_ms_p50']:.0f} ms "
                        f"p99 {s['ttft_ms_p99']:.0f} ms, errors {s.get('num_errors', 0):.0f} "
                        f"({(time.time() - t) / 60:.1f} min)")
                time.sleep(args.cooldown)
        except Exception as e:  # one bad configuration must not end the night
            log(f"{g.key}: ERROR {type(e).__name__}: {e}")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
            server_log.close()
            time.sleep(10)  # let the driver release GPU memory before the next server

    log(f"sweep finished: {len(summary)} runs completed this session")


if __name__ == "__main__":
    main()
