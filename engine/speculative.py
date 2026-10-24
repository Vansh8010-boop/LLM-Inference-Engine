"""Speculative decoding: a small draft model proposes k tokens, the target
model checks all of them in one forward pass.

Each round:
  1. The draft decodes k tokens autoregressively (cheap: 0.5B model).
  2. The target runs ONE forward pass over [pending tokens + k drafts] and
     gets its own next-token distribution at every draft position.
  3. accept_tokens() keeps the longest acceptable prefix of the drafts, then
     adds one token from the target (a correction, or a bonus if all k passed),
     so every round yields between 1 and k + 1 tokens.
  4. Both KV caches are rolled back to the accepted length. With a contiguous
     cache that is just moving the position counter: stale slots get
     overwritten by the next round.

With greedy decoding the result is the target's own greedy text. With
sampling, the acceptance rule (Leviathan et al. 2023) makes the output
distribution exactly the target's.
"""

from dataclasses import dataclass
from typing import Iterator

import torch

from engine.generate import SamplingParams
from engine.kv_cache import ContiguousKVCache
from engine.model import Transformer
from engine.sampler import probs_from_logits


@dataclass
class SpecStats:
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    generated: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_round(self) -> float:
        return self.generated / self.rounds if self.rounds else 0.0


def accept_tokens(
    drafts: list[int],
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    temperature: float,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[int, int]:
    """Decide how many drafts to keep and which token comes after them.

    drafts:        k proposed token ids
    draft_probs:   (k, V) draft distributions they were sampled from (None if greedy)
    target_logits: (k + 1, V) target logits at each draft position plus one more
    Returns (num_accepted, next_token).
    """
    k = len(drafts)
    if temperature <= 0:
        target_choice = target_logits.argmax(dim=-1).tolist()
        m = 0
        while m < k and drafts[m] == target_choice[m]:
            m += 1
        return m, target_choice[m]

    q = probs_from_logits(target_logits, temperature, top_p)  # (k + 1, V)
    for i, d in enumerate(drafts):
        # Accept with probability min(1, q(d) / p(d)).
        u = torch.rand((), device=q.device, generator=generator)
        if u * draft_probs[i, d] > q[i, d]:
            # Rejected: sample from the leftover mass max(0, q - p), renormalized.
            residual = (q[i] - draft_probs[i]).clamp(min=0)
            residual = residual / residual.sum() if residual.sum() > 0 else q[i]
            return i, torch.multinomial(residual, 1, generator=generator).item()
    return k, torch.multinomial(q[k], 1, generator=generator).item()


class SpeculativeDecoder:
    def __init__(self, target: Transformer, draft: Transformer, eos_ids: set[int], k: int = 4):
        if target.config.vocab_size != draft.config.vocab_size:
            raise ValueError("draft and target must share a vocabulary")
        self.target, self.draft, self.eos_ids, self.k = target, draft, eos_ids, k

    @torch.inference_mode()
    def generate_stream(self, prompt_ids: list[int], params: SamplingParams,
                        stats: SpecStats | None = None) -> Iterator[int]:
        stats = stats if stats is not None else SpecStats()
        dev = self.target.device
        greedy = params.temperature <= 0
        generator = torch.Generator(device=dev).manual_seed(params.seed) if params.seed is not None else None
        cap = len(prompt_ids) + params.max_tokens + self.k + 2
        t_cache = ContiguousKVCache(self.target.config, 1, cap, dev, self.target.dtype)
        d_cache = ContiguousKVCache(self.draft.config, 1, cap, dev, self.draft.dtype)

        seq = list(prompt_ids)
        t_pos = d_pos = 0  # tokens of seq whose K/V each model has cached
        produced = 0
        while produced < params.max_tokens:
            # Never draft past max_tokens: a round yields up to k_now + 1 tokens.
            k_now = min(self.k, params.max_tokens - produced - 1)

            # 1. Draft k_now tokens.
            drafts, draft_probs = [], []
            if k_now:
                logits = self.draft(torch.tensor([seq[d_pos:]], device=dev), d_pos, d_cache)[0, -1]
                d_pos = len(seq)
                for i in range(k_now):
                    if greedy:
                        token = logits.argmax().item()
                    else:
                        p = probs_from_logits(logits, params.temperature, params.top_p)
                        token = torch.multinomial(p, 1, generator=generator).item()
                        draft_probs.append(p)
                    drafts.append(token)
                    if i < k_now - 1:
                        logits = self.draft(torch.tensor([[token]], device=dev), d_pos, d_cache)[0, -1]
                        d_pos += 1

            # 2. Target scores the pending tokens and every draft in one pass.
            inp = seq[t_pos:] + drafts
            target_logits = self.target(torch.tensor([inp], device=dev), t_pos, t_cache,
                                        keep_last=k_now + 1)[0]

            # 3. Accept a prefix of the drafts, plus one target token.
            m, extra = accept_tokens(drafts, torch.stack(draft_probs) if draft_probs else None,
                                     target_logits, params.temperature, params.top_p, generator)

            # 4. Roll both caches back to what survived.
            t_pos = len(seq) + m
            d_pos = min(d_pos, len(seq) + m)
            stats.rounds += 1
            stats.proposed += k_now
            stats.accepted += m

            for token in drafts[:m] + [extra]:
                seq.append(token)
                produced += 1
                stats.generated += 1
                yield token
                if (token in self.eos_ids and not params.ignore_eos) or produced >= params.max_tokens:
                    return

    def generate(self, prompt_ids: list[int], params: SamplingParams, stats: SpecStats | None = None) -> list[int]:
        return list(self.generate_stream(prompt_ids, params, stats))
