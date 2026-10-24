"""Our engine vs Hugging Face: greedy output must match token-for-token.

Run after every change:
    pytest tests/test_correctness.py -v

Quick run while iterating (fewer prompts and tokens):
    $env:MINI_INFER_TEST_PROMPTS=10; $env:MINI_INFER_TEST_TOKENS=16; pytest tests -v
"""

import gc

import pytest
import torch

from engine.generate import LLM, SamplingParams
from tests.conftest import MAX_NEW_TOKENS, MODEL_DIR, requires_gpu

pytestmark = requires_gpu


@pytest.fixture(scope="module")
def ours(llm) -> LLM:
    return llm


@pytest.fixture(scope="module")
def hf(ours: LLM):
    from transformers import AutoModelForCausalLM, GenerationConfig

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16, attn_implementation="sdpa")
    model = model.to(ours.device).eval()
    # Drop the checkpoint's repetition_penalty/top_k defaults: pure greedy only.
    model.generation_config = GenerationConfig(
        eos_token_id=sorted(ours.eos_ids), pad_token_id=ours.tokenizer.pad_token_id
    )
    yield model
    del model  # free the GPU for the batched tests that follow
    gc.collect()
    torch.cuda.empty_cache()


def hf_greedy(hf, prompt_ids: list[int], max_new: int) -> list[int]:
    ids = torch.tensor([prompt_ids], device=hf.device)
    out = hf.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new, do_sample=False)
    return out[0, len(prompt_ids):].tolist()


def test_prefill_logits_close(ours: LLM, hf, prompts):
    """Every position's logits should agree closely, and the argmax almost always."""
    for prompt_ids in prompts[:5]:
        ids = torch.tensor([prompt_ids], device=ours.device)
        with torch.inference_mode():
            ref = hf(ids).logits.float()
        got = ours.model(ids, last_only=False).float()
        max_diff = (ref - got).abs().max().item()
        agree = (ref.argmax(-1) == got.argmax(-1)).float().mean().item()
        assert max_diff < 0.5, f"max logit diff {max_diff:.4f} on a {len(prompt_ids)}-token prompt"
        assert agree > 0.98, f"argmax agreement only {agree:.2%}"


def test_greedy_matches_hf(ours: LLM, hf, prompts):
    """Both KV cache layouts must reproduce Hugging Face exactly."""
    params = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0)
    mismatches = []
    for i, prompt_ids in enumerate(prompts):
        expected = hf_greedy(hf, prompt_ids, MAX_NEW_TOKENS)
        for kv_cache in ("paged", "contiguous"):
            got = ours.generate(prompt_ids, params, kv_cache=kv_cache)
            if got != expected:
                at = next((j for j, (a, b) in enumerate(zip(got, expected)) if a != b), min(len(got), len(expected)))
                mismatches.append(f"prompt {i} ({len(prompt_ids)} tok, {kv_cache}): diverges at token {at}")
    assert not mismatches, f"{len(mismatches)} mismatches over {len(prompts)} prompts:\n" + "\n".join(mismatches)
    assert ours.pool.allocator.num_free == ours.pool.allocator.num_blocks, "paged generation leaked blocks"


def test_cached_matches_uncached(ours: LLM, prompts):
    params = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0)
    for prompt_ids in prompts[:5]:
        assert ours.generate(prompt_ids, params, kv_cache="paged") == ours.generate(prompt_ids, params, kv_cache="none")


def test_sampling_is_seeded(ours: LLM, prompts):
    params = SamplingParams(max_tokens=16, temperature=0.8, top_p=0.9, seed=1234)
    assert ours.generate(prompts[0], params) == ours.generate(prompts[0], params)
