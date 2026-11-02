"""Continuous batching vs the single-sequence engine.

The single-sequence path is bit-identical to Hugging Face (test_correctness),
so it is the reference here. Batched execution cannot be bit-identical: a
matmul over 32 rows picks different cuBLAS kernels than one over 1 row, and
masked attention uses a different SDPA kernel. Those differ in the last fp16
bit, which can flip a greedy choice only when the top two logits are tied to
within a few fp16 steps. So: every sequence must match exactly, except that a
divergence is allowed at a genuine near-tie, and the reference's logit gap
there is checked.
"""

import gc

import pytest
import torch

from engine.generate import LLM, SamplingParams
from engine.llm_engine import LLMEngine
from tests.conftest import MAX_NEW_TOKENS, requires_gpu

pytestmark = requires_gpu

TIE_TOLERANCE = 0.125  # 8 fp16 steps at logit magnitude ~16; real bugs show gaps of 5-20
MIN_EXACT = 0.9


def near_tie_gap(llm: LLM, prompt: list[int], ref: list[int], got: list[int]) -> tuple[int, float] | None:
    """Where got first differs from ref, and the reference's logit gap between the two tokens."""
    j = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), None)
    if j is None:
        return None if len(ref) == len(got) else (min(len(ref), len(got)), float("inf"))
    logits = llm.model(torch.tensor([prompt + ref[:j]], device=llm.device))[0, -1].float()
    return j, abs(logits[ref[j]] - logits[got[j]]).item()


def check_against_reference(llm: LLM, prompts, got, params) -> None:
    exact, problems = 0, []
    for i, (prompt, out) in enumerate(zip(prompts, got)):
        ref = llm.generate(prompt, params)
        div = near_tie_gap(llm, prompt, ref, out)
        if div is None:
            exact += 1
        elif div[1] > TIE_TOLERANCE:
            problems.append(f"prompt {i}: diverges at token {div[0]} with logit gap {div[1]:.3f}")
    assert not problems, "divergences that are not near-ties:\n" + "\n".join(problems)
    assert exact / len(prompts) >= MIN_EXACT, f"only {exact}/{len(prompts)} exact matches"


@pytest.fixture(scope="module")
def engine(llm: LLM):
    eng = LLMEngine(llm.model, llm.tokenizer, llm.eos_ids, kv_cache_gb=0.5)
    yield eng
    del eng  # give the KV pool back before later modules load more weights
    gc.collect()
    torch.cuda.empty_cache()


def test_batched_matches_reference(llm: LLM, engine: LLMEngine, prompts):
    params = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0)
    got = engine.generate(prompts, params)  # all submitted at once
    check_against_reference(llm, prompts, got, params)
    assert engine.pool.allocator.num_free == engine.pool.allocator.num_blocks, "leaked KV blocks"


def test_preemption_recomputes_correctly(llm: LLM, prompts):
    short = [p for p in prompts if len(p) < 150][:16]
    params = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0, ignore_eos=True)
    # A pool with a third of the blocks the requests need in total (but room for the
    # largest one), so the scheduler must evict and recompute.
    needed = [-(-(len(p) + MAX_NEW_TOKENS) // 16) for p in short]
    num_blocks = max(max(needed), sum(needed) // 3)
    tiny = LLMEngine(llm.model, llm.tokenizer, llm.eos_ids, kv_cache_gb=num_blocks * 16 * 12288 / 1e9)
    got = tiny.generate(short, params)
    assert tiny.scheduler.total_preemptions > 0, "pool was large enough that nothing was preempted"
    assert all(len(g) == MAX_NEW_TOKENS for g in got)
    check_against_reference(llm, short, got, params)
    assert tiny.pool.allocator.num_free == tiny.pool.allocator.num_blocks


def test_mixed_sampling_params(engine: LLMEngine, prompts):
    """Greedy and sampled requests in one batch; seeded sampling is reproducible."""
    greedy = SamplingParams(max_tokens=16)
    sampled = SamplingParams(max_tokens=16, temperature=0.8, top_p=0.9, seed=7)
    seqs = [engine.add_request(prompts[0], greedy), engine.add_request(prompts[1], sampled)]
    while engine.has_unfinished():
        engine.step()
    again = engine.generate([prompts[1]], sampled)[0]
    assert seqs[1].output_ids == again
    assert all(s.finish_reason in ("stop", "length") for s in seqs)
