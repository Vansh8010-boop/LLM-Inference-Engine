"""INT8 weight-only quantization with per-output-channel scales.

For each linear layer W (out x in):
    scale[o]  = max_i |W[o, i]| / 127
    W_q[o, i] = round(W[o, i] / scale[o])  in [-127, 127], stored as int8
    y         = (x @ W_qᵀ) * scale + bias

Activations stay fp16. The matmul uses PyTorch's fused weight-only kernel
(torch._weight_int8pack_mm), which reads the int8 weights directly; if that op
is unavailable it falls back to dequantizing to fp16 per call (correct, slower).
"""

import torch
import torch.nn.functional as F
from torch import nn

QUANTIZED_LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def quantize_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(out, in) float weight -> (int8 weight, per-row scale in the weight's dtype)."""
    w = weight.float()
    scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale.to(weight.dtype)


def _fused_int8_available(device: torch.device) -> bool:
    try:
        x = torch.zeros(1, 16, device=device, dtype=torch.float16)
        torch._weight_int8pack_mm(x, torch.zeros(16, 16, device=device, dtype=torch.int8),
                                  torch.ones(16, device=device, dtype=torch.float16))
        return True
    except (RuntimeError, AttributeError, NotImplementedError):
        return False


class Int8Linear(nn.Module):
    def __init__(self, linear: nn.Linear, fused: bool):
        super().__init__()
        q, scale = quantize_per_channel(linear.weight.data)
        self.register_buffer("weight", q)
        self.register_buffer("scale", scale)
        self.bias = linear.bias
        self.fused = fused
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            y = torch._weight_int8pack_mm(x.reshape(-1, self.in_features), self.weight, self.scale)
            y = y.view(*x.shape[:-1], self.out_features)
        else:
            y = F.linear(x, self.weight.to(x.dtype)) * self.scale
        return y + self.bias if self.bias is not None else y

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, int8, fused={self.fused}"


def quantize_model(model: nn.Module) -> nn.Module:
    """Swap every transformer-layer projection for an Int8Linear, in place."""
    device = next(model.parameters()).device
    fused = _fused_int8_available(device)
    for layer in model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in QUANTIZED_LINEARS:
                if isinstance(getattr(parent, name, None), nn.Linear):
                    setattr(parent, name, Int8Linear(getattr(parent, name), fused))
    if device.type == "cuda":
        torch.cuda.empty_cache()  # release the replaced fp16 weights
    return model


def weight_bytes(model: nn.Module) -> int:
    """Bytes of parameters + buffers, counting tied tensors once (excludes RoPE tables)."""
    seen, total = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        if t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            total += t.numel() * t.element_size()
    return total
