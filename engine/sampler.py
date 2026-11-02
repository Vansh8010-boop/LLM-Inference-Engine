"""Next-token selection: greedy, temperature and top-p (nucleus) sampling."""

import torch


def probs_from_logits(logits: torch.Tensor, temperature: float, top_p: float = 1.0) -> torch.Tensor:
    """(..., V) logits -> the fp32 distribution that sample() draws from (temperature > 0)."""
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        # Drop a token once the tokens ranked above it already cover top_p.
        # The top token always survives, since nothing is ranked above it.
        cum_before = sorted_probs.cumsum(dim=-1) - sorted_probs
        sorted_probs[cum_before >= top_p] = 0.0
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def sample(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """logits: (B, V) -> next token ids (B,). temperature 0 means greedy."""
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probs = probs_from_logits(logits, temperature, top_p)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
