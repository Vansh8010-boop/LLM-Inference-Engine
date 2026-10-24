"""Block allocator and paged KV pool: fast CPU tests, no model weights needed."""

import pytest
import torch

from engine.kv_cache import BlockAllocator, ContiguousKVCache, OutOfBlocksError, PagedKVPool
from engine.model import ModelConfig

TINY = ModelConfig(
    vocab_size=32, hidden_size=16, intermediate_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
    head_dim=4, rms_norm_eps=1e-6, rope_theta=10000.0, max_position_embeddings=128, tie_word_embeddings=True,
)


def test_grow_allocates_blocks_only_at_boundaries():
    alloc = BlockAllocator(num_blocks=8, block_size=4)
    assert alloc.grow(0, 1) == [0]
    assert alloc.grow(0, 4) == [0]       # still fits in block 0
    assert alloc.grow(0, 5) == [0, 1]    # crosses into a second block
    assert alloc.num_free == 6
    assert alloc.stats()["reserved_tokens"] == 8 and alloc.stats()["used_tokens"] == 5


def test_free_returns_blocks_for_reuse():
    alloc = BlockAllocator(num_blocks=4, block_size=4)
    alloc.grow(0, 8)
    alloc.grow(1, 8)
    assert not alloc.can_grow(2, 1)
    alloc.free(0)
    assert alloc.num_free == 2
    assert sorted(alloc.grow(2, 8)) == [0, 1]  # seq 0's blocks are reused


def test_out_of_blocks_raises_and_changes_nothing():
    alloc = BlockAllocator(num_blocks=2, block_size=4)
    alloc.grow(0, 4)
    with pytest.raises(OutOfBlocksError):
        alloc.grow(1, 9)
    assert alloc.num_free == 1
    assert 1 not in alloc.tables


def test_paged_cache_returns_same_kv_as_contiguous():
    """Interleave two sequences so their blocks are not adjacent, then compare to a plain buffer."""
    torch.manual_seed(0)
    pool = PagedKVPool(TINY, num_blocks=16, block_size=4, device="cpu", dtype=torch.float32)
    caches = [pool.sequence(0), pool.sequence(1)]
    refs = [ContiguousKVCache(TINY, 1, 32, "cpu", torch.float32) for _ in range(2)]
    steps = [(0, 6), (6, 1), (7, 1), (8, 5), (13, 1)]  # prefill 6, decode, decode, chunk of 5, decode

    for start, t in steps:
        for cache, ref in zip(caches, refs):
            for layer in range(TINY.num_layers):
                k = torch.randn(1, TINY.num_kv_heads, t, TINY.head_dim)
                v = torch.randn_like(k)
                got_k, got_v = cache.update(layer, k, v, start)
                exp_k, exp_v = ref.update(layer, k, v, start)
                assert torch.equal(got_k, exp_k) and torch.equal(got_v, exp_v)

    assert pool.allocator.tables[0] != list(range(4)), "blocks should be interleaved between sequences"
    for cache in caches:
        cache.free()
    assert pool.allocator.num_free == 16
