"""Build frozen benchmark workloads in bench/workloads/.

Every benchmark in the project reads one of these JSONL files, so all
configurations see identical requests. Each line is:
    {"prompt": str, "prompt_tokens": int, "max_tokens": int}

Usage:
    python -m bench.make_workloads sharegpt                 # 1,000 first-turn prompts
    python -m bench.make_workloads synthetic --prompt-len 512 --output-len 128 -n 200
"""

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKLOADS_DIR = PROJECT_ROOT / "bench" / "workloads"
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_TOKENIZER = PROJECT_ROOT / "models" / "Qwen2.5-0.5B-Instruct"

SHAREGPT_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"

# Same filters as vLLM's benchmark_serving.py, so numbers are comparable.
MIN_TOKENS = 4
MAX_PROMPT_TOKENS = 1024
MAX_TOTAL_TOKENS = 2048

WORDS = (
    "system model token memory cache block batch request user server latency "
    "throughput attention layer weight tensor kernel queue schedule decode prefill "
    "page table stream engine sample greedy vector matrix bandwidth compute"
).split()


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    avg_in = sum(r["prompt_tokens"] for r in rows) / len(rows)
    avg_out = sum(r["max_tokens"] for r in rows) / len(rows)
    print(f"[saved] {path} ({len(rows)} requests, avg prompt {avg_in:.0f} tok, avg output {avg_out:.0f} tok)")


def make_sharegpt(tokenizer, n: int, seed: int) -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        SHAREGPT_REPO, SHAREGPT_FILE, repo_type="dataset", local_dir=DATA_DIR / "sharegpt"
    )
    with open(path, encoding="utf-8") as f:
        conversations = json.load(f)

    # First human turn as the prompt, the real reply length as max_tokens.
    pairs = [
        (c["conversations"][0]["value"], c["conversations"][1]["value"])
        for c in conversations
        if len(c.get("conversations", [])) >= 2
        and c["conversations"][0].get("from") == "human"
        and c["conversations"][1].get("from") == "gpt"
    ]
    random.Random(seed).shuffle(pairs)

    rows = []
    for prompt, reply in pairs:
        prompt_tokens = len(tokenizer.encode(prompt))
        output_tokens = len(tokenizer.encode(reply))
        if prompt_tokens < MIN_TOKENS or output_tokens < MIN_TOKENS:
            continue
        if prompt_tokens > MAX_PROMPT_TOKENS or prompt_tokens + output_tokens > MAX_TOTAL_TOKENS:
            continue
        rows.append({"prompt": prompt, "prompt_tokens": prompt_tokens, "max_tokens": output_tokens})
        if len(rows) == n:
            break
    return rows


def make_synthetic(tokenizer, n: int, prompt_len: int, output_len: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        text = " ".join(rng.choice(WORDS) for _ in range(prompt_len * 2))
        prompt = tokenizer.decode(tokenizer.encode(text)[:prompt_len])
        rows.append(
            {"prompt": prompt, "prompt_tokens": len(tokenizer.encode(prompt)), "max_tokens": output_len}
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen benchmark workloads")
    sub = parser.add_subparsers(dest="kind", required=True)

    sg = sub.add_parser("sharegpt")
    sg.add_argument("-n", type=int, default=1000)

    syn = sub.add_parser("synthetic")
    syn.add_argument("-n", type=int, default=200)
    syn.add_argument("--prompt-len", type=int, default=512)
    syn.add_argument("--output-len", type=int, default=128)

    for p in (sg, syn):
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
        p.add_argument("--out", type=Path, help="output path (default: bench/workloads/<name>.jsonl)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.kind == "sharegpt":
        rows = make_sharegpt(tokenizer, args.n, args.seed)
        out = args.out or WORKLOADS_DIR / f"sharegpt_{args.n}.jsonl"
    else:
        rows = make_synthetic(tokenizer, args.n, args.prompt_len, args.output_len, args.seed)
        out = args.out or WORKLOADS_DIR / f"synthetic_p{args.prompt_len}_o{args.output_len}_n{args.n}.jsonl"
    write_jsonl(rows, out)


if __name__ == "__main__":
    main()
