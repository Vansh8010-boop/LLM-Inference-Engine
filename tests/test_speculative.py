"""Speculative decoding: the acceptance rule (CPU) and the full decoder (GPU, needs 1.5B)."""

import gc

import pytest
import torch

from engine.generate import LLM, SamplingParams
from engine.speculative import SpecStats, SpeculativeDecoder, accept_tokens
from tests.conftest import PROJECT_ROOT, requires_gpu

TARGET_DIR = PROJECT_ROOT / "models" / "Qwen2.5-1.5B-Instruct"


def test_greedy_accepts_matching_prefix():
    logits = torch.full((4, 5), -10.0)
    for pos, tok in enumerate([1, 2, 3, 4]):
        logits[pos, tok] = 10.0  # target's choices: 1, 2, 3, then 4
    assert accept_tokens([1, 2, 0], None, logits, temperature=0) == (2, 3)   # third draft wrong -> correction 3
    assert accept_tokens([1, 2, 3], None, logits, temperature=0) == (3, 4)   # all accepted -> bonus 4
    assert accept_tokens([], None, logits[:1], temperature=0) == (0, 1)      # k = 0 is plain decoding


def test_sampling_rule_reproduces_target_distribution():
    """Whatever the draft proposes, the first emitted token must follow the target's distribution."""
    gen = torch.Generator().manual_seed(0)
    q = torch.tensor([0.5, 0.2, 0.2, 0.1])   # target
    p = torch.tensor([0.1, 0.1, 0.4, 0.4])   # a deliberately bad draft
    target_logits = q.log().repeat(2, 1)     # probs_from_logits(log q, T=1) == q
    counts = torch.zeros(4)
    trials = 40_000
    for _ in range(trials):
        d = torch.multinomial(p, 1, generator=gen).item()
        m, extra = accept_tokens([d], p[None], target_logits, temperature=1.0, generator=gen)
        counts[d if m == 1 else extra] += 1
    assert torch.allclose(counts / trials, q, atol=0.01), counts / trials


@pytest.fixture(scope="module")
def target(llm):
    if not TARGET_DIR.exists():
        pytest.skip("models/Qwen2.5-1.5B-Instruct not downloaded")
    gc.collect()
    torch.cuda.empty_cache()
    model = LLM(TARGET_DIR, device=llm.device, kv_cache_gb=0.1)
    yield model
    del model
    gc.collect()
    torch.cuda.empty_cache()


@requires_gpu
def test_greedy_speculative_matches_target(llm, target, prompts):
    """0.5B drafts, 1.5B verifies: greedy output must be the 1.5B model's own (up to fp16 near-ties)."""
    from tests.test_batched import check_against_reference

    spec = SpeculativeDecoder(target.model, llm.model, target.eos_ids, k=4)
    params = SamplingParams(max_tokens=32, temperature=0.0)
    chosen = [p for p in prompts if len(p) < 400][:20]
    stats = SpecStats()
    got = [spec.generate(p, params, stats) for p in chosen]
    assert stats.accepted > 0, "draft never accepted: the models are not being compared correctly"
    check_against_reference(target, chosen, got, params)


@requires_gpu
def test_speculative_respects_max_tokens_and_seed(llm, target, prompts):
    spec = SpeculativeDecoder(target.model, llm.model, target.eos_ids, k=6)
    params = SamplingParams(max_tokens=13, temperature=0.8, top_p=0.9, seed=3, ignore_eos=True)
    a = spec.generate(prompts[0], params)
    assert len(a) == 13
    assert a == spec.generate(prompts[0], params)
