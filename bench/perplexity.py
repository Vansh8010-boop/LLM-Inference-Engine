"""WikiText-2 perplexity with our own forward pass (quality check for quantization).

Sliding window, as in the Hugging Face perplexity guide: windows of
max_length tokens advance by stride, and each token is scored once, with up
to max_length - stride tokens of context before it.

Usage:
    python -m bench.perplexity                      # 0.5B, fp16
    python -m bench.perplexity --quantize int8
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from engine.device import setup_device
from engine.generate import DEFAULT_MODEL
from engine.model import Transformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def wikitext2_ids(tokenizer) -> list[int]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", cache_dir=str(DATA_DIR / "hf-datasets"))
    return tokenizer.encode("\n\n".join(ds["text"]))


@torch.inference_mode()
def perplexity(model: Transformer, ids: list[int], max_length: int = 1024, stride: int = 512,
               max_tokens: int | None = None) -> float:
    ids_t = torch.tensor(ids[:max_tokens] if max_tokens else ids, device=model.device)
    n = ids_t.numel()
    nll, count, prev_end = 0.0, 0, 0
    for begin in range(0, n - 1, stride):
        end = min(begin + max_length, n)
        window = ids_t[begin:end][None]
        logits = model(window, last_only=False)[0]  # (T, V) fp16
        # Score only tokens not scored by the previous window; logits[t] predicts token t+1.
        first = max(prev_end, begin + 1) - begin
        for a in range(first, end - begin, 256):  # chunk to bound the fp32 copy
            b = min(a + 256, end - begin)
            nll += F.cross_entropy(logits[a - 1 : b - 1].float(), window[0, a:b], reduction="sum").item()
            count += b - a
        prev_end = end
        if end == n:
            break
    return math.exp(nll / count)


def main() -> None:
    parser = argparse.ArgumentParser(description="WikiText-2 perplexity")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--quantize", choices=["int8"], default=None)
    parser.add_argument("--max-tokens", type=int, default=None, help="evaluate only the first N tokens")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Transformer.from_pretrained(args.model, setup_device(), quantize=args.quantize)
    ids = wikitext2_ids(tokenizer)
    print(f"[ppl] {Path(args.model).name} {args.quantize or 'fp16'}: "
          f"{perplexity(model, ids, max_tokens=args.max_tokens):.3f} over {args.max_tokens or len(ids)} tokens")


if __name__ == "__main__":
    main()
