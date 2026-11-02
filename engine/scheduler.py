"""Continuous batching scheduler.

Every engine step the scheduler rebuilds the batch:
  1. Running sequences each need one more KV slot for their next token. When
     the block pool runs out, the newest running sequence is preempted: its
     blocks are freed and it goes back to the front of the waiting queue, to be
     recomputed (prompt + tokens generated so far re-prefilled) later.
  2. If nothing had to be preempted, waiting requests are admitted first-come
     first-served while there are free blocks (above a small watermark), seats
     (max_num_seqs) and prefill token budget (max_prefill_tokens).
  3. The engine runs one forward pass for decodes + prefills together, then
     finished sequences are retired and their blocks freed.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import torch

from engine.kv_cache import BlockAllocator


class SeqStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(eq=False)
class Sequence:
    seq_id: int
    prompt_len: int
    token_ids: list[int]                # prompt followed by generated tokens
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    ignore_eos: bool = False
    generator: torch.Generator | None = None
    arrival: float = field(default_factory=time.monotonic)
    status: SeqStatus = SeqStatus.WAITING
    num_preemptions: int = 0
    finish_reason: str | None = None

    @property
    def output_ids(self) -> list[int]:
        return self.token_ids[self.prompt_len:]

    @property
    def num_output(self) -> int:
        return len(self.token_ids) - self.prompt_len


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 64
    max_prefill_tokens: int = 2048
    watermark: float = 0.01  # keep this fraction of blocks free when admitting


@dataclass
class ScheduledStep:
    decode: list[Sequence]
    prefill: list[Sequence]
    preempted: list[Sequence]


class Scheduler:
    def __init__(self, config: SchedulerConfig, allocator: BlockAllocator):
        self.config = config
        self.allocator = allocator
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.watermark_blocks = int(config.watermark * allocator.num_blocks)
        self.total_preemptions = 0

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> ScheduledStep:
        # 1. A slot for every running sequence's next token, oldest first.
        preempted: list[Sequence] = []
        queue = sorted(self.running, key=lambda s: s.arrival)
        decode: list[Sequence] = []
        while queue:
            seq = queue.pop(0)
            while not self.allocator.can_grow(seq.seq_id, len(seq.token_ids)):
                victim = queue.pop() if queue else seq  # newest left in line
                self._preempt(victim, preempted)
                if victim is seq:
                    break
            else:
                self.allocator.grow(seq.seq_id, len(seq.token_ids))
                decode.append(seq)
        self.running = decode

        # 2. Admit new work only when no one was just evicted.
        prefill: list[Sequence] = []
        budget = self.config.max_prefill_tokens
        while not preempted and self.waiting and len(self.running) + len(prefill) < self.config.max_num_seqs:
            seq = self.waiting[0]
            n = len(seq.token_ids)
            if n > budget and prefill:  # an oversized prompt still runs, but alone
                break
            free_after = self.allocator.num_free - (self.allocator.blocks_needed(n) - len(self.allocator.tables.get(seq.seq_id, [])))
            if free_after < self.watermark_blocks:
                break
            self.allocator.grow(seq.seq_id, n)
            self.waiting.popleft()
            seq.status = SeqStatus.RUNNING
            prefill.append(seq)
            budget -= n
        self.running = decode + prefill  # new list: `decode` is returned below and must not grow
        return ScheduledStep(decode, prefill, preempted)

    def _preempt(self, seq: Sequence, preempted: list[Sequence]) -> None:
        """Recompute-style preemption: drop the KV cache, keep the tokens."""
        self.allocator.free(seq.seq_id)
        seq.status = SeqStatus.WAITING
        seq.num_preemptions += 1
        self.total_preemptions += 1
        self.waiting.appendleft(seq)
        preempted.append(seq)

    def finish(self, seq: Sequence, reason: str) -> None:
        self.allocator.free(seq.seq_id)
        seq.status = SeqStatus.FINISHED
        seq.finish_reason = reason
        if seq in self.running:
            self.running.remove(seq)

    def abort_all(self, reason: str = "abort") -> list[Sequence]:
        aborted = list(self.running) + list(self.waiting)
        for seq in aborted:
            self.finish(seq, reason)
        self.waiting.clear()
        return aborted
