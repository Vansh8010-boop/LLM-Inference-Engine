"""INT8 weight-only quantization."""

import gc

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from engine.quant import Int8Linear, quantize_per_channel, weight_bytes
from tests.conftest import MODEL_DIR, requires_gpu


def test_per_channel_roundtrip_error_is_within_half_a_step():
    torch.manual_seed(0)
    w = torch.randn(64, 32) * torch.linspace(0.01, 5, 64)[:, None]  # rows on very different scales
    q, scale = quantize_per_channel(w)
    assert q.dtype == torch.int8 and q.abs().max() <= 127
    err = (q.float() * scale[:, None] - w).abs()
    assert (err <= scale[:, None] / 2 + 1e-6).all()  # per-row scales keep small rows precise


@requires_gpu
def test_fused_kernel_matches_dequantized_matmul():
    torch.manual_seed(0)
    lin = nn.Linear(896, 4864, bias=True).cuda().half()
    x = torch.randn(3, 7, 896, device="cuda", dtype=torch.float16)
    fused, plain = Int8Linear(lin, fused=True), Int8Linear(lin, fused=False)
    expected = F.linear(x.float(), plain.weight.float() * plain.scale.float()[:, None], lin.bias.float())
    for layer in (fused, plain):
        assert torch.allclose(layer(x).float(), expected, atol=0.05, rtol=0.01)


@requires_gpu
def test_int8_model_stays_close_to_fp16(llm, prompts):
    from engine.model import Transformer

    int8 = Transformer.from_pretrained(MODEL_DIR, llm.device, quantize="int8")
    try:
        assert weight_bytes(int8) < 0.7 * weight_bytes(llm.model)
        for p in prompts[:3]:
            ids = torch.tensor([p], device=llm.device)
            ref = llm.model(ids, last_only=False)[0].float()
            got = int8(ids, last_only=False)[0].float()
            agree = (ref.argmax(-1) == got.argmax(-1)).float().mean().item()
            assert agree > 0.9, f"INT8 top-1 agreement with FP16 only {agree:.1%}"
    finally:
        del int8
        gc.collect()
        torch.cuda.empty_cache()
