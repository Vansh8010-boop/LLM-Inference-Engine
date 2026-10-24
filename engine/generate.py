"""Single-sequence generation loop on top of engine.model.

Prefill runs the whole prompt once and fills the KV cache; every decode step
then feeds just the newest token. kv_cache selects the storage:
    "paged"       blocks from the shared PagedKVPool (default)
    "contiguous"  a private max-length buffer per request (week 1)
    "none"        re-run the full sequence every step (week 1 "before" chart)

Try it:
    python -m engine.generate "Explain the KV cache in one paragraph." --chat --max-tokens 128
"""

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import torch
from transformers import AutoTokenizer

from engine.device import setup_device
from engine.kv_cache import ContiguousKVCache, PagedKVPool
from engine.model import Transformer

KVCacheKind = Literal["paged", "contiguous", "none"]
_seq_ids = itertools.count()
from engine.sampler import sample

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "Qwen2.5-0.5B-Instruct"


@dataclass
class SamplingParams:
    max_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    ignore_eos: bool = False
    seed: int | None = None


def load_eos_ids(model_dir: Path) -> set[int]:
    for name in ("generation_config.json", "config.json"):
        path = model_dir / name
        if path.exists():
            eos = json.loads(path.read_text()).get("eos_token_id")
            if eos is not None:
                return set(eos if isinstance(eos, list) else [eos])
    return set()


@torch.inference_mode()
def generate_stream(
    model: Transformer,
    prompt_ids: list[int],
    params: SamplingParams,
    eos_ids: set[int],
    kv_cache: KVCacheKind = "paged",
    pool: PagedKVPool | None = None,
) -> Iterator[int]:
    """Yields new token ids one at a time; the EOS token is yielded before stopping."""
    device = model.device
    ids = torch.tensor([prompt_ids], device=device)
    generator = None
    if params.seed is not None:
        generator = torch.Generator(device=device).manual_seed(params.seed)

    cache = None
    if kv_cache == "paged":
        if pool is None:
            raise ValueError("kv_cache='paged' needs a PagedKVPool")
        cache = pool.sequence(next(_seq_ids))
    elif kv_cache == "contiguous":
        cache = ContiguousKVCache(model.config, 1, len(prompt_ids) + params.max_tokens, device, model.dtype)

    try:
        logits = model(ids, 0, cache)  # prefill
        pos = len(prompt_ids)
        for i in range(params.max_tokens):
            token = sample(logits[:, -1], params.temperature, params.top_p, generator)
            token_id = token.item()
            yield token_id
            if (token_id in eos_ids and not params.ignore_eos) or i == params.max_tokens - 1:
                return
            if cache is not None:
                logits = model(token.view(1, 1), pos, cache)
                pos += 1
            else:
                ids = torch.cat([ids, token.view(1, 1)], dim=1)
                logits = model(ids)
    finally:
        if kv_cache == "paged":
            cache.free()  # blocks go back to the pool, even if the caller stops early


class LLM:
    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL,
        device=None,
        dtype=torch.float16,
        kv_cache_gb: float = 1.0,
        block_size: int = 16,
    ):
        self.model_dir = Path(model_dir)
        self.device = device or setup_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = Transformer.from_pretrained(self.model_dir, self.device, dtype)
        self.eos_ids = load_eos_ids(self.model_dir)
        self.pool = PagedKVPool.from_budget(
            self.model.config, int(kv_cache_gb * 1e9), block_size, self.device, dtype
        )

    def encode(self, prompt: str, chat: bool = False) -> list[int]:
        if chat:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True,
                return_dict=False,
            )
        return self.tokenizer.encode(prompt)

    def stream(self, prompt_ids: list[int], params: SamplingParams, kv_cache: KVCacheKind = "paged") -> Iterator[int]:
        return generate_stream(self.model, prompt_ids, params, self.eos_ids, kv_cache, self.pool)

    def generate(self, prompt_ids: list[int], params: SamplingParams, kv_cache: KVCacheKind = "paged") -> list[int]:
        return list(self.stream(prompt_ids, params, kv_cache))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with the mini-infer engine")
    parser.add_argument("prompt")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--chat", action="store_true", help="wrap the prompt in the chat template")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--kv-cache", choices=["paged", "contiguous", "none"], default="paged")
    args = parser.parse_args()

    llm = LLM(args.model)
    params = SamplingParams(args.max_tokens, args.temperature, args.top_p, seed=args.seed)
    ids: list[int] = []
    printed = ""
    for token_id in llm.stream(llm.encode(args.prompt, args.chat), params, kv_cache=args.kv_cache):
        ids.append(token_id)
        text = llm.tokenizer.decode(ids, skip_special_tokens=True)
        if not text.endswith("�"):
            sys.stdout.write(text[len(printed):])
            sys.stdout.flush()
            printed = text
    print()


if __name__ == "__main__":
    main()
