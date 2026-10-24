import json
import os
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "Qwen2.5-0.5B-Instruct"
WORKLOAD = PROJECT_ROOT / "bench" / "workloads" / "sharegpt_1000.jsonl"
N_PROMPTS = int(os.environ.get("MINI_INFER_TEST_PROMPTS", 100))
MAX_NEW_TOKENS = int(os.environ.get("MINI_INFER_TEST_TOKENS", 32))

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or not MODEL_DIR.exists(),
    reason="needs a CUDA GPU and models/Qwen2.5-0.5B-Instruct",
)


@pytest.fixture(scope="session")
def llm():
    """Our single-sequence engine, shared by every GPU test (one copy of the weights)."""
    from engine.generate import LLM

    return LLM(MODEL_DIR, kv_cache_gb=0.5)


@pytest.fixture(scope="session")
def prompts(llm) -> list[list[int]]:
    if WORKLOAD.exists():
        with WORKLOAD.open(encoding="utf-8") as f:
            texts = [json.loads(line)["prompt"] for _, line in zip(range(N_PROMPTS), f)]
    else:
        texts = [f"Write a short note about topic number {i}." for i in range(N_PROMPTS)]
    return [llm.tokenizer.encode(t) for t in texts]
