"""Download Qwen2.5 weights + tokenizer into the project's models/ folder.

Usage:
    python scripts/download_model.py                              # Qwen2.5-0.5B-Instruct
    python scripts/download_model.py --model Qwen/Qwen2.5-1.5B-Instruct
    python scripts/download_model.py --model Qwen/Qwen2.5-0.5B --force

Each model ends up in models/<name>/ (config.json, model.safetensors,
tokenizer files), ready for AutoModelForCausalLM.from_pretrained or for
loading the safetensors directly in engine/model.py.
"""

import argparse
import shutil
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
# Temporary download cache inside the project, so nothing lands on C:.
CACHE_DIR = MODELS_DIR / ".download-cache"


def download(repo_id: str, force: bool) -> Path:
    out_dir = MODELS_DIR / repo_id.split("/")[-1]
    if out_dir.exists() and not force:
        print(f"[skip] {out_dir} already exists (use --force to re-download)")
        return out_dir

    print(f"[download] {repo_id} -> {out_dir}")
    start = time.time()
    # Fetch the checkpoint files straight to disk (never loading the weights into RAM),
    # reusing anything an interrupted earlier run left in the cache.
    snapshot = Path(snapshot_download(
        repo_id, cache_dir=CACHE_DIR,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "tokenizer*", "merges*", "vocab*"],
    ))
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in snapshot.iterdir():
        shutil.copyfile(f.resolve(), out_dir / f.name)

    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    size_gb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e9
    print(f"[done] {size_gb:.2f} GB in {time.time() - start:.0f}s")
    return out_dir


def smoke_test(model_dir: Path) -> None:
    """Load from the local folder only and run one greedy generation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=dtype, local_files_only=True
    ).to(device)

    messages = [{"role": "user", "content": "Who are you?"}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    outputs = model.generate(**inputs, max_new_tokens=48, do_sample=False)
    reply = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )
    print(f"[smoke test on {device}] {reply.strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        action="append",
        help="Hugging Face repo id; repeat for several (default: Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--no-test", action="store_true", help="skip the generation smoke test")
    args = parser.parse_args()

    for repo_id in args.model or ["Qwen/Qwen2.5-0.5B-Instruct"]:
        model_dir = download(repo_id, args.force)
        if not args.no_test:
            smoke_test(model_dir)


if __name__ == "__main__":
    main()
