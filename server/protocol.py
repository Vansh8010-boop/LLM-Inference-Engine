"""OpenAI-compatible /v1/completions plumbing shared by every server.

A backend only has to implement submit(Request): run the request somewhere
(usually a GPU thread) and push events back through Request.send():
    ("token", token_id)                     one generated token
    ("done", finish_reason, prompt_tokens)  "stop" or "length"
    ("error", message)
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from engine.device import git_commit, gpu_info


class CompletionRequest(BaseModel):
    prompt: str
    model: str | None = None
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int | None = None
    stream: bool = False
    # Generate exactly max_tokens; benchmarks use this so output lengths match the workload.
    ignore_eos: bool = False


@dataclass
class Request:
    params: CompletionRequest
    loop: asyncio.AbstractEventLoop
    events: asyncio.Queue = field(default_factory=asyncio.Queue)

    def send(self, *event) -> None:
        """Thread-safe: called from the GPU thread."""
        self.loop.call_soon_threadsafe(self.events.put_nowait, event)


def create_app(title: str, submit: Callable[[Request], None], tokenizer, model_name: str,
               info: Callable[[], dict]) -> FastAPI:
    app = FastAPI(title=title)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/info")
    def server_info():
        mem = {}
        if torch.cuda.is_available():
            mem = {"memory_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
                   "max_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)}
        return {"model": model_name, "commit": git_commit(), **gpu_info(), **mem, **info()}

    async def token_stream(req: Request):
        """Yields (text_delta, finish_reason, usage) as tokens arrive."""
        ids: list[int] = []
        sent = ""
        while True:
            event = await req.events.get()
            if event[0] == "error":
                raise HTTPException(status_code=500, detail=event[1])
            if event[0] == "done":
                text = tokenizer.decode(ids, skip_special_tokens=True)
                usage = {"prompt_tokens": event[2], "completion_tokens": len(ids),
                         "total_tokens": event[2] + len(ids)}
                yield text[len(sent):], event[1], usage
                return
            ids.append(event[1])
            text = tokenizer.decode(ids, skip_special_tokens=True)
            if text.endswith("�"):  # partial UTF-8 character, wait for the next token
                continue
            yield text[len(sent):], None, None
            sent = text

    def chunk(rid: str, text: str, finish_reason, usage=None) -> dict:
        body = {"id": rid, "object": "text_completion", "created": int(time.time()), "model": model_name,
                "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason}]}
        if usage:
            body["usage"] = usage
        return body

    @app.post("/v1/completions")
    async def completions(params: CompletionRequest):
        if params.max_tokens < 1:
            raise HTTPException(status_code=400, detail="max_tokens must be >= 1")
        req = Request(params=params, loop=asyncio.get_running_loop())
        submit(req)
        rid = f"cmpl-{uuid.uuid4().hex[:24]}"

        if params.stream:
            async def sse():
                try:
                    async for text, finish_reason, usage in token_stream(req):
                        yield f"data: {json.dumps(chunk(rid, text, finish_reason, usage))}\n\n"
                except HTTPException as e:
                    yield f"data: {json.dumps({'error': {'message': e.detail}})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        parts, finish_reason, usage = [], None, None
        async for text, finish_reason, usage in token_stream(req):
            parts.append(text)
        return chunk(rid, "".join(parts), finish_reason, usage)

    return app
