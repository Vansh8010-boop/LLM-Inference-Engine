"""Scheduler policy: fast CPU tests with a real BlockAllocator and no model."""

from engine.kv_cache import BlockAllocator
from engine.scheduler import Scheduler, SchedulerConfig, Sequence, SeqStatus


def make_seq(seq_id: int, prompt_len: int, arrival: float | None = None) -> Sequence:
    seq = Sequence(seq_id=seq_id, prompt_len=prompt_len, token_ids=[1] * prompt_len, max_tokens=100)
    seq.arrival = float(seq_id) if arrival is None else arrival
    return seq


def decode_one(sched: Scheduler, seqs: list[Sequence]) -> None:
    """Stand-in for the engine: every scheduled sequence gains one token."""
    for s in seqs:
        s.token_ids.append(2)


def test_admits_fcfs_up_to_max_num_seqs():
    sched = Scheduler(SchedulerConfig(max_num_seqs=2, watermark=0.0), BlockAllocator(100, 4))
    seqs = [make_seq(i, 4) for i in range(3)]
    for s in seqs:
        sched.add(s)
    step = sched.schedule()
    assert step.prefill == seqs[:2] and step.decode == []
    assert list(sched.waiting) == [seqs[2]]


def test_new_prefills_are_not_also_decoded():
    sched = Scheduler(SchedulerConfig(watermark=0.0), BlockAllocator(100, 4))
    a, b = make_seq(0, 4), make_seq(1, 4)
    sched.add(a)
    decode_one(sched, sched.schedule().prefill)
    sched.add(b)
    step = sched.schedule()
    assert step.decode == [a] and step.prefill == [b]
    assert sched.running == [a, b]


def test_prefill_token_budget_limits_admission():
    sched = Scheduler(SchedulerConfig(max_prefill_tokens=10, watermark=0.0), BlockAllocator(100, 4))
    for i in range(3):
        sched.add(make_seq(i, 6))
    assert len(sched.schedule().prefill) == 1  # 6 + 6 > 10


def test_oversized_prompt_runs_alone():
    sched = Scheduler(SchedulerConfig(max_prefill_tokens=10, watermark=0.0), BlockAllocator(100, 4))
    sched.add(make_seq(0, 50))
    assert len(sched.schedule().prefill) == 1


def test_preempts_newest_when_blocks_run_out():
    # 4 blocks of 4 tokens; two sequences of 8 tokens fill the pool exactly.
    sched = Scheduler(SchedulerConfig(watermark=0.0), BlockAllocator(4, 4))
    old, new = make_seq(0, 8), make_seq(1, 8)
    sched.add(old)
    sched.add(new)
    running = sched.schedule().prefill
    decode_one(sched, running)  # both now need a 9th slot = a third block each

    step = sched.schedule()
    assert step.preempted == [new]
    assert step.decode == [old] and step.prefill == []  # no admissions on a preemption step
    assert new.status is SeqStatus.WAITING and sched.waiting[0] is new
    assert new.seq_id not in sched.allocator.tables  # its KV blocks were freed
    assert len(new.token_ids) == 9  # tokens kept for recompute


def test_preempted_sequence_is_recomputed_when_room_frees_up():
    sched = Scheduler(SchedulerConfig(watermark=0.0), BlockAllocator(4, 4))
    old, new = make_seq(0, 8), make_seq(1, 8)
    sched.add(old)
    sched.add(new)
    decode_one(sched, sched.schedule().prefill)
    sched.schedule()  # preempts `new`
    sched.finish(old, "length")

    step = sched.schedule()
    assert step.prefill == [new]  # re-prefilled with prompt + generated token
    assert sched.allocator.lengths[new.seq_id] == 9


def test_finish_frees_blocks():
    alloc = BlockAllocator(10, 4)
    sched = Scheduler(SchedulerConfig(watermark=0.0), alloc)
    s = make_seq(0, 8)
    sched.add(s)
    sched.schedule()
    sched.finish(s, "stop")
    assert alloc.num_free == 10 and not sched.has_unfinished()
