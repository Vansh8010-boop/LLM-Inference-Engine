"""KV cache storage.

Week 1: ContiguousKVCache preallocates max_seq_len slots for every sequence,
the simplest correct design. Its weakness: a request that stops at 50 tokens
still holds memory for max_seq_len.

Week 2: paged KV cache, like OS virtual memory.
    BlockAllocator  bookkeeping only: a free list of fixed-size blocks and a
                    block table per sequence (logical block i -> physical block)
    PagedKVPool     one big K/V tensor per layer, carved into those blocks
    PagedKVCache    one sequence's view of the pool; same update() interface
                    as ContiguousKVCache, so the model doesn't know the difference

A sequence only ever holds ceil(len / block_size) blocks, so the waste is at
most block_size - 1 slots per sequence instead of max_seq_len - len.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from engine.model import ModelConfig, repeat_kv


def kv_bytes_per_token(cfg: ModelConfig, dtype: torch.dtype = torch.float16) -> int:
    """K and V, for every layer and KV head: 2 * layers * kv_heads * head_dim * bytes.

    Qwen2.5-0.5B: 2 * 24 * 2 * 64 * 2 B = 12 KiB per token.
    """
    return 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * torch.finfo(dtype).bits // 8


class ContiguousKVCache:
    def __init__(self, cfg: ModelConfig, batch_size: int, max_seq_len: int, device, dtype=torch.float16):
        shape = (cfg.num_layers, batch_size, cfg.num_kv_heads, max_seq_len, cfg.head_dim)
        self.k = torch.empty(shape, device=device, dtype=dtype)
        self.v = torch.empty(shape, device=device, dtype=dtype)
        self.max_seq_len = max_seq_len

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor, start_pos: int):
        """Write (B, KV, T, D) at positions start_pos.., return K/V for positions 0..start_pos+T."""
        end = start_pos + k.shape[2]
        if end > self.max_seq_len:
            raise ValueError(f"KV cache full: need {end} slots, have {self.max_seq_len}")
        self.k[layer, :, :, start_pos:end] = k
        self.v[layer, :, :, start_pos:end] = v
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    @property
    def nbytes(self) -> int:
        return self.k.nbytes + self.v.nbytes


class OutOfBlocksError(RuntimeError):
    pass


class BlockAllocator:
    """Which physical blocks each sequence owns. Pure Python, no tensors."""

    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks - 1, -1, -1))  # pop() hands out 0, 1, 2, ...
        self.tables: dict[int, list[int]] = {}
        self.lengths: dict[int, int] = {}

    def blocks_needed(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)

    def can_grow(self, seq_id: int, num_tokens: int) -> bool:
        extra = self.blocks_needed(num_tokens) - len(self.tables.get(seq_id, []))
        return extra <= self.num_free

    def grow(self, seq_id: int, num_tokens: int) -> list[int]:
        """Make room for num_tokens in total; registers the sequence on first call."""
        if not self.can_grow(seq_id, num_tokens):
            extra = self.blocks_needed(num_tokens) - len(self.tables.get(seq_id, []))
            raise OutOfBlocksError(f"sequence {seq_id} needs {extra} more blocks, {self.num_free} free")
        table = self.tables.setdefault(seq_id, [])
        for _ in range(self.blocks_needed(num_tokens) - len(table)):
            table.append(self.free_blocks.pop())
        self.lengths[seq_id] = max(num_tokens, self.lengths.get(seq_id, 0))
        return table

    def free(self, seq_id: int) -> None:
        self.free_blocks.extend(reversed(self.tables.pop(seq_id, [])))
        self.lengths.pop(seq_id, None)

    def stats(self) -> dict:
        used = sum(self.lengths.values())
        reserved = sum(len(t) for t in self.tables.values()) * self.block_size
        return {"sequences": len(self.tables), "used_tokens": used, "reserved_tokens": reserved,
                "free_blocks": self.num_free, "waste_pct": 100 * (reserved - used) / reserved if reserved else 0.0}


class PagedKVPool:
    """K/V storage for every layer, shaped (layers, num_blocks * block_size, KV, D).

    Physical block b owns the flat slots [b * block_size, (b + 1) * block_size).
    """

    def __init__(self, cfg: ModelConfig, num_blocks: int, block_size: int, device, dtype=torch.float16):
        shape = (cfg.num_layers, num_blocks * block_size, cfg.num_kv_heads, cfg.head_dim)
        self.k = torch.empty(shape, device=device, dtype=dtype)
        self.v = torch.empty(shape, device=device, dtype=dtype)
        self.allocator = BlockAllocator(num_blocks, block_size)
        self.block_size = block_size
        self.device = torch.device(device)

    @classmethod
    def from_budget(cls, cfg: ModelConfig, budget_bytes: int, block_size: int, device, dtype=torch.float16):
        num_blocks = budget_bytes // (kv_bytes_per_token(cfg, dtype) * block_size)
        return cls(cfg, num_blocks, block_size, device, dtype)

    def sequence(self, seq_id: int) -> "PagedKVCache":
        return PagedKVCache(self, seq_id)

    @property
    def nbytes(self) -> int:
        return self.k.nbytes + self.v.nbytes


class PagedKVCache:
    """One sequence's view of the pool. Batched paged attention arrives with the week 3 scheduler."""

    def __init__(self, pool: PagedKVPool, seq_id: int):
        self.pool = pool
        self.seq_id = seq_id
        self._table_gpu: torch.Tensor | None = None
        self._read_slots: torch.Tensor | None = None
        self._write_slots: torch.Tensor | None = None

    def _prepare(self, start_pos: int, end: int) -> None:
        """Once per forward pass (layer 0): grab blocks and map positions -> flat slots."""
        table = self.pool.allocator.grow(self.seq_id, end)
        if self._table_gpu is None or len(self._table_gpu) != len(table):
            self._table_gpu = torch.tensor(table, device=self.pool.device)
        bs = self.pool.block_size
        pos = torch.arange(end, device=self.pool.device)
        self._read_slots = self._table_gpu[pos // bs] * bs + pos % bs
        self._write_slots = self._read_slots[start_pos:end]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor, start_pos: int):
        """Write (1, KV, T, D) at positions start_pos.., gather K/V for 0..start_pos+T through the block table."""
        if k.shape[0] != 1:
            raise ValueError("PagedKVCache holds a single sequence")
        if layer == 0:
            self._prepare(start_pos, start_pos + k.shape[2])
        k_pool, v_pool = self.pool.k[layer], self.pool.v[layer]
        k_pool[self._write_slots] = k[0].transpose(0, 1)  # (T, KV, D)
        v_pool[self._write_slots] = v[0].transpose(0, 1)
        # Gather (correct before fast): (S, KV, D) -> (1, KV, S, D)
        keys = k_pool[self._read_slots].transpose(0, 1).unsqueeze(0)
        values = v_pool[self._read_slots].transpose(0, 1).unsqueeze(0)
        return keys, values

    def free(self) -> None:
        self.pool.allocator.free(self.seq_id)


@dataclass
class PagedBatch:
    """Layout of one continuous-batching step over the shared pool.

    The N new tokens are flattened in this order: one token per decode
    sequence first, then each prefill sequence's whole prompt.
        positions      (N,)  absolute position of each token (for RoPE)
        slot_mapping   (N,)  flat pool slot each token's K/V is written to
        decode_slots   (Bd, S) pool slots to gather for each decode sequence,
                             padded to the longest one with slot 0
        decode_mask    (Bd, 1, 1, S) True where a slot is real; None if Bd <= 1
        prefill_ranges token ranges [a, b) of each prefill sequence
        logits_index   (num_seqs,) index of each sequence's last token
    """

    pool: PagedKVPool
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    num_decode: int
    decode_slots: torch.Tensor | None
    decode_mask: torch.Tensor | None
    prefill_ranges: list[tuple[int, int]]
    logits_index: torch.Tensor
    batched = True

    @classmethod
    def build(cls, pool: PagedKVPool, decode: list[tuple[int, int]], prefill: list[tuple[int, int]]) -> "PagedBatch":
        """decode / prefill: (seq_id, total_tokens) with blocks already allocated.

        A decode sequence feeds its newest token (position total-1); a prefill
        sequence feeds all of its tokens from position 0.
        """
        bs = pool.block_size
        tables = pool.allocator.tables
        positions, slots, last = [], [], []
        for seq_id, total in decode:
            pos = total - 1
            positions.append(pos)
            slots.append(tables[seq_id][pos // bs] * bs + pos % bs)
            last.append(len(positions) - 1)
        prefill_ranges = []
        for seq_id, total in prefill:
            table = tables[seq_id]
            start = len(positions)
            positions.extend(range(total))
            slots.extend(table[p // bs] * bs + p % bs for p in range(total))
            prefill_ranges.append((start, len(positions)))
            last.append(len(positions) - 1)

        dev = pool.device
        decode_slots = decode_mask = None
        if decode:
            lengths = torch.tensor([total for _, total in decode], device=dev)
            max_len = max(total for _, total in decode)
            max_blocks = -(-max_len // bs)
            padded = [tables[s][:max_blocks] + [0] * (max_blocks - len(tables[s][:max_blocks])) for s, _ in decode]
            table_t = torch.tensor(padded, device=dev)
            pos = torch.arange(max_len, device=dev)
            decode_slots = table_t[:, pos // bs] * bs + pos % bs  # (Bd, S)
            if len(decode) > 1:
                decode_mask = (pos[None, :] < lengths[:, None])[:, None, None, :]
        return cls(
            pool=pool,
            positions=torch.tensor(positions, device=dev),
            slot_mapping=torch.tensor(slots, device=dev),
            num_decode=len(decode),
            decode_slots=decode_slots,
            decode_mask=decode_mask,
            prefill_ranges=prefill_ranges,
            logits_index=torch.tensor(last, device=dev),
        )

    def attend(self, layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """q: (1, H, N, D), k/v: (1, KV, N, D) -> attention output (1, H, N, D)."""
        k_pool, v_pool = self.pool.k[layer], self.pool.v[layer]
        k_pool[self.slot_mapping] = k[0].transpose(0, 1)
        v_pool[self.slot_mapping] = v[0].transpose(0, 1)
        outs = []

        bd = self.num_decode
        if bd:
            qd = q[0, :, :bd].transpose(0, 1).unsqueeze(2)       # (Bd, H, 1, D)
            kd = k_pool[self.decode_slots].permute(0, 2, 1, 3)   # (Bd, KV, S, D)
            vd = v_pool[self.decode_slots].permute(0, 2, 1, 3)
            if self.decode_mask is None:
                od = F.scaled_dot_product_attention(qd, kd, vd, enable_gqa=True)
            else:
                n_rep = q.shape[1] // k.shape[1]
                od = F.scaled_dot_product_attention(
                    qd, repeat_kv(kd, n_rep), repeat_kv(vd, n_rep), attn_mask=self.decode_mask
                )
            outs.append(od.squeeze(2).transpose(0, 1).unsqueeze(0))  # (1, H, Bd, D)

        # Prefills start at position 0, so their K/V are exactly this step's tokens.
        for a, b in self.prefill_ranges:
            outs.append(F.scaled_dot_product_attention(
                q[:, :, a:b], k[:, :, a:b], v[:, :, a:b], is_causal=True, enable_gqa=True
            ))
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)
