"""Hugging Face baseline: an OpenAI-compatible server around model.generate().

This is the week 0 yardstick every later optimization is measured against.
    --max-batch-size 1   one request at a time
    --max-batch-size N   naive static batching: wait briefly, grab up to N
                         queued requests, pad them together, run generate()
                         until the *longest* one finishes, then take the next batch

Run:
    python -m server.baseline_server --max-batch-size 8
Try it:
    curl http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" \
         -d '{"prompt": "The capital of France is", "max_tokens": 20, "temperature": 0, "stream": true}'
"""

import argparse
import queue
import threading
import time
from pathlib import Path

import torch
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation.streamers import BaseStreamer

from engine.device import setup_device
from server.protocol import Request, create_app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "Qwen2.5-0.5B-Instruct"


class BatchStreamer(BaseStreamer):
    """Forwards each decode step's tokens to the requests in a static batch.

    generate() calls put() once with the prompt ids, then once per step with a
    (batch,) tensor of new tokens. Each request stops on its own EOS or
    max_tokens, even though generate() keeps running for the whole batch.
    """

    def __init__(self, requests: list[Request], prompt_tokens: list[int], eos_ids: set[int]):
        self.requests = requests
        self.prompt_tokens = prompt_tokens
        self.eos_ids = eos_ids
        self.counts = [0] * len(requests)
        self.finished = [False] * len(requests)
        self.saw_prompt = False

    def _finish(self, i: int, reason: str) -> None:
        self.finished[i] = True
        self.requests[i].send("done", reason, self.prompt_tokens[i])

    def put(self, value: torch.Tensor) -> None:
        if not self.saw_prompt:
            self.saw_prompt = True
            return
        for i, token in enumerate(value.view(-1).tolist()):
            if self.finished[i]:
                continue
            req = self.requests[i]
            if token in self.eos_ids and not req.params.ignore_eos:
                self._finish(i, "stop")
                continue
            req.send("token", token)
            self.counts[i] += 1
            if self.counts[i] >= req.params.max_tokens:
                self._finish(i, "length")

    def end(self) -> None:
        for i in range(len(self.requests)):
            if not self.finished[i]:
                self._finish(i, "length")


class BaselineEngine:
    def __init__(self, model_path: str, max_batch_size: int, batch_wait_ms: float, gpu_fraction: float | None):
        self.device = setup_device(gpu_fraction)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(self.device).eval()

        # Replace the checkpoint's sampling defaults (top_k, repetition_penalty, ...)
        # so only the parameters a request sends are applied, same as our engine.
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        self.model.generation_config = GenerationConfig(
            eos_token_id=list(self.eos_ids), pad_token_id=self.tokenizer.pad_token_id
        )

        self.model_name = Path(model_path).name
        self.max_batch_size = max_batch_size
        self.batch_wait_s = batch_wait_ms / 1000
        self.queue: queue.Queue[Request] = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, req: Request) -> None:
        self.queue.put(req)

    def _worker(self) -> None:
        while True:
            batch = [self.queue.get()]
            deadline = time.monotonic() + self.batch_wait_s
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.queue.get(timeout=remaining))
                except queue.Empty:
                    break
            self._run_batch(batch)

    @torch.inference_mode()
    def _run_batch(self, batch: list[Request]) -> None:
        try:
            enc = self.tokenizer(
                [r.params.prompt for r in batch], return_tensors="pt", padding=True
            ).to(self.device)
            prompt_tokens = enc["attention_mask"].sum(dim=1).tolist()
            # Static batching can only use one set of sampling params: take the first request's.
            p = batch[0].params
            sampling = (
                {"do_sample": True, "temperature": p.temperature, "top_p": p.top_p}
                if p.temperature > 0
                else {"do_sample": False}
            )
            max_new = max(r.params.max_tokens for r in batch)
            if all(r.params.ignore_eos for r in batch):
                sampling["min_new_tokens"] = max_new
            self.model.generate(
                **enc,
                max_new_tokens=max_new,
                streamer=BatchStreamer(batch, prompt_tokens, self.eos_ids),
                **sampling,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            for r in batch:
                r.send("error", "CUDA out of memory; lower --max-batch-size or raise --gpu-fraction")
        except Exception as e:  # keep the worker alive for the next batch
            for r in batch:
                r.send("error", f"{type(e).__name__}: {e}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Hugging Face baseline server")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--max-batch-size", type=int, default=1, help="1 = sequential, >1 = static batching")
    parser.add_argument("--batch-wait-ms", type=float, default=20, help="how long to wait to fill a batch")
    parser.add_argument("--gpu-fraction", type=float, default=None, help="cap on GPU memory (default 0.75)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    engine = BaselineEngine(args.model, args.max_batch_size, args.batch_wait_ms, args.gpu_fraction)
    config = {"max_batch_size": args.max_batch_size, "batch_wait_ms": args.batch_wait_ms,
              "gpu_fraction": args.gpu_fraction, "dtype": str(engine.model.dtype)}
    print(f"[server] {engine.model_name}, max batch {args.max_batch_size}, "
          f"listening on http://{args.host}:{args.port}", flush=True)
    app = create_app("mini-infer HF baseline", engine.submit, engine.tokenizer, engine.model_name,
                     info=lambda: {"engine": "hf-baseline", "config": config})
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
