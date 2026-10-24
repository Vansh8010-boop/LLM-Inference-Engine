"""The serving engine: scheduler + paged KV pool + model, one step at a time.

    engine = LLMEngine.from_pretrained("models/Qwen2.5-0.5B-Instruct")
    seq = engine.add_request(prompt_ids, SamplingParams(max_tokens=64))
    while engine.has_unfinished():
        for out in engine.step():
            ...  # out.seq, out.token_id, out.finish_reason
"""

import itertools
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer

from engine.device import setup_device
from engine.generate import DEFAULT_MODEL, SamplingParams, load_eos_ids
from engine.kv_cache import PagedBatch, PagedKVPool
from engine.model import Transformer
from engine.sampler import sample
from engine.scheduler import Scheduler, SchedulerConfig, Sequence


@dataclass
class StepOutput:
    seq: Sequence
    token_id: int
    finish_reason: str | None  # "stop" (EOS), "length" (max_tokens), or None while running


class RequestTooLongError(ValueError):
    pass


class LLMEngine:
    def __init__(
        self,
        model: Transformer,
        tokenizer,
        eos_ids: set[int],
        kv_cache_gb: float = 2.0,
        block_size: int = 16,
        scheduler_config: SchedulerConfig | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.eos_ids = eos_ids
        self.pool = PagedKVPool.from_budget(model.config, int(kv_cache_gb * 1e9), block_size, model.device, model.dtype)
        self.scheduler = Scheduler(scheduler_config or SchedulerConfig(), self.pool.allocator)
        self._ids = itertools.count()
        self.num_steps = 0

    @classmethod
    def from_pretrained(cls, model_dir: str | Path = DEFAULT_MODEL, device=None, dtype=torch.float16, **kwargs):
        model_dir = Path(model_dir)
        device = device or setup_device()
        model = Transformer.from_pretrained(model_dir, device, dtype)
        return cls(model, AutoTokenizer.from_pretrained(model_dir), load_eos_ids(model_dir), **kwargs)

    # ---- requests ----
    def add_request(self, prompt_ids: list[int], params: SamplingParams) -> Sequence:
        total = len(prompt_ids) + params.max_tokens
        alloc = self.pool.allocator
        if alloc.blocks_needed(total) > alloc.num_blocks or total > self.model.config.max_position_embeddings:
            raise RequestTooLongError(f"prompt + max_tokens = {total} tokens does not fit in the KV cache")
        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=self.model.device).manual_seed(params.seed)
        seq = Sequence(
            seq_id=next(self._ids), prompt_len=len(prompt_ids), token_ids=list(prompt_ids),
            max_tokens=params.max_tokens, temperature=params.temperature, top_p=params.top_p,
            ignore_eos=params.ignore_eos, generator=generator,
        )
        self.scheduler.add(seq)
        return seq

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    def abort_all(self) -> list[Sequence]:
        return self.scheduler.abort_all()

    # ---- one iteration ----
    @torch.inference_mode()
    def step(self) -> list[StepOutput]:
        sched = self.scheduler.schedule()
        seqs = sched.decode + sched.prefill
        if not seqs:
            return []

        batch = PagedBatch.build(
            self.pool,
            decode=[(s.seq_id, len(s.token_ids)) for s in sched.decode],
            prefill=[(s.seq_id, len(s.token_ids)) for s in sched.prefill],
        )
        input_ids = [s.token_ids[-1] for s in sched.decode]
        for s in sched.prefill:
            input_ids.extend(s.token_ids)
        logits = self.model.forward_batch(torch.tensor(input_ids, device=self.model.device), batch)
        next_tokens = self._sample(seqs, logits)
        self.num_steps += 1

        outputs = []
        for seq, token in zip(seqs, next_tokens):
            seq.token_ids.append(token)
            reason = None
            if token in self.eos_ids and not seq.ignore_eos:
                reason = "stop"
            elif seq.num_output >= seq.max_tokens:
                reason = "length"
            if reason:
                self.scheduler.finish(seq, reason)
            outputs.append(StepOutput(seq, token, reason))
        return outputs

    def _sample(self, seqs: list[Sequence], logits: torch.Tensor) -> list[int]:
        if all(s.temperature <= 0 for s in seqs):
            return logits.argmax(dim=-1).tolist()  # one GPU sync for the whole batch
        tokens = [sample(logits[i : i + 1], s.temperature, s.top_p, s.generator) for i, s in enumerate(seqs)]
        return torch.cat(tokens).tolist()

    # ---- offline use ----
    def generate(self, prompts: list[list[int]], params: SamplingParams) -> list[list[int]]:
        """Run every prompt to completion with continuous batching; returns generated ids per prompt."""
        seqs = [self.add_request(p, params) for p in prompts]
        while self.has_unfinished():
            self.step()
        return [s.output_ids for s in seqs]

    def stats(self) -> dict:
        return {"steps": self.num_steps, "running": len(self.scheduler.running),
                "waiting": len(self.scheduler.waiting), "preemptions": self.scheduler.total_preemptions,
                **self.pool.allocator.stats()}
