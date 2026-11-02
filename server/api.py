"""mini-infer engine server: OpenAI-compatible /v1/completions with streaming.

One GPU thread owns the LLMEngine. Each loop iteration it pulls newly arrived
requests into the scheduler, runs one engine step (every running sequence
advances one token, new ones are prefilled), and streams the new tokens back
to their HTTP handlers. Requests join and leave the batch at every step.

Run:
    python -m server.api
    python -m server.api --kv-cache-gb 2.5 --max-num-seqs 64
"""

import argparse
import queue
import threading
from pathlib import Path

import uvicorn

from engine.device import setup_device
from engine.generate import DEFAULT_MODEL, SamplingParams
from engine.llm_engine import LLMEngine, RequestTooLongError
from engine.scheduler import SchedulerConfig
from server.protocol import Request, create_app


class EngineWorker:
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self.inbox: queue.Queue[Request] = queue.Queue()
        self.active: dict[int, Request] = {}  # seq_id -> request
        threading.Thread(target=self._loop, daemon=True, name="engine").start()

    def submit(self, req: Request) -> None:
        self.inbox.put(req)

    def _admit(self, req: Request) -> None:
        p = req.params
        prompt_ids = self.engine.tokenizer.encode(p.prompt)
        params = SamplingParams(max_tokens=p.max_tokens, temperature=p.temperature, top_p=p.top_p,
                                ignore_eos=p.ignore_eos, seed=p.seed)
        try:
            seq = self.engine.add_request(prompt_ids, params)
        except RequestTooLongError as e:
            req.send("error", str(e))
            return
        self.active[seq.seq_id] = req

    def _loop(self) -> None:
        while True:
            if not self.engine.has_unfinished():
                self._admit(self.inbox.get())  # idle: block until work arrives
            while True:
                try:
                    self._admit(self.inbox.get_nowait())
                except queue.Empty:
                    break
            try:
                outputs = self.engine.step()
            except Exception as e:  # report to every caller, free all blocks, keep serving
                for seq in self.engine.abort_all():
                    if (req := self.active.pop(seq.seq_id, None)) is not None:
                        req.send("error", f"{type(e).__name__}: {e}")
                continue
            for out in outputs:
                req = self.active.get(out.seq.seq_id)
                if req is None:
                    continue
                if out.finish_reason != "stop":  # the EOS token itself is not streamed
                    req.send("token", out.token_id)
                if out.finish_reason:
                    req.send("done", out.finish_reason, out.seq.prompt_len)
                    del self.active[out.seq.seq_id]


def main() -> None:
    parser = argparse.ArgumentParser(description="mini-infer OpenAI-compatible server")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--kv-cache-gb", type=float, default=2.0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-prefill-tokens", type=int, default=2048)
    parser.add_argument("--quantize", choices=["int8"], default=None, help="INT8 weight-only quantization")
    parser.add_argument("--gpu-fraction", type=float, default=None, help="cap on GPU memory (default 0.75)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    engine = LLMEngine.from_pretrained(
        args.model, device=setup_device(args.gpu_fraction), quantize=args.quantize, kv_cache_gb=args.kv_cache_gb,
        block_size=args.block_size,
        scheduler_config=SchedulerConfig(args.max_num_seqs, args.max_prefill_tokens),
    )
    worker = EngineWorker(engine)
    config = {k: v for k, v in vars(args).items() if k not in ("host", "port")}
    model_name = Path(args.model).name
    app = create_app("mini-infer", worker.submit, engine.tokenizer, model_name,
                     info=lambda: {"engine": "mini-infer", "config": config, "stats": engine.stats()})
    alloc = engine.pool.allocator
    print(f"[server] {model_name}: {alloc.num_blocks} KV blocks x {alloc.block_size} tokens "
          f"({engine.pool.nbytes / 1e9:.2f} GB), max {args.max_num_seqs} seqs, "
          f"listening on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
