"""Our own Qwen2 transformer forward pass.

Hugging Face is used only to *store* the weights: they are read straight from
the safetensors file into the modules below. Op order and dtypes mirror the
reference implementation (RMSNorm in fp32, RoPE tables in fp32 cast to the
model dtype, SDPA attention) so greedy decoding matches it token-for-token.

Shapes used throughout:
    B = batch, T = new tokens this step, S = total cached length
    H = query heads (14), KV = key/value heads (2), D = head dim (64)
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "ModelConfig":
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        rope = cfg.get("rope_parameters") or {}
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_layers=cfg["num_hidden_layers"],
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"],
            rms_norm_eps=cfg["rms_norm_eps"],
            rope_theta=rope.get("rope_theta", cfg.get("rope_theta", 10000.0)),
            max_position_embeddings=cfg["max_position_embeddings"],
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def rope_tables(head_dim: int, max_pos: int, theta: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin for every position, (max_pos, D), kept in fp32."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64, device=device).float() / head_dim))
    pos = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, heads, T, D); cos/sin: (T, D) for this step's absolute positions.
    return x * cos + rotate_half(x) * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, KV, S, D) -> (B, KV * n_rep, S, D): each KV head serves n_rep query heads."""
    if n_rep == 1:
        return x
    b, kv, s, d = x.shape
    return x[:, :, None].expand(b, kv, n_rep, s, d).reshape(b, kv * n_rep, s, d)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.num_heads // cfg.num_kv_heads
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=True)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin, cache, start_pos: int) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if getattr(cache, "batched", False):
            # Continuous batching: many sequences flattened into T tokens; the batch
            # object writes K/V into the paged pool and runs each sequence's attention.
            out = cache.attend(self.layer_idx, q, k, v)
            return self.o_proj(out.transpose(1, 2).reshape(b, t, self.num_heads * self.head_dim))

        if cache is not None:
            # Store this step's K/V, get back everything cached so far: (B, KV, S, D).
            k, v = cache.update(self.layer_idx, k, v, start_pos)
        # Without a mask, SDPA handles GQA itself (enable_gqa) and picks a fused kernel;
        # this is also what Hugging Face does, which keeps the rounding identical.
        if t == 1:
            # One new token attends to every cached position: no mask needed.
            out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        elif start_pos == 0:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            # T new tokens after start_pos cached ones: query i sees keys <= start_pos + i.
            # Masked GQA falls back to the slow math kernel, so expand KV heads first.
            s = k.shape[2]
            mask = torch.ones(t, s, dtype=torch.bool, device=x.device).tril(diagonal=start_pos)
            out = F.scaled_dot_product_attention(
                q, repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep), attn_mask=mask
            )

        out = out.transpose(1, 2).reshape(b, t, self.num_heads * self.head_dim)
        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg, layer_idx)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cache, start_pos: int) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache, start_pos)
        return x + self.mlp(self.post_attention_layernorm(x))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.rope_cos: torch.Tensor | None = None
        self.rope_sin: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.embed_tokens.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embed_tokens.weight.dtype

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0, cache=None, last_only: bool = True):
        """input_ids: (B, T) tokens at positions start_pos .. start_pos+T-1.

        Without a cache, input_ids must be the whole sequence (start_pos=0).
        Returns logits (B, 1, V) for the last position, or (B, T, V) if last_only=False.
        """
        t = input_ids.shape[1]
        x = self.embed_tokens(input_ids)
        cos = self.rope_cos[start_pos : start_pos + t].to(x.dtype)
        sin = self.rope_sin[start_pos : start_pos + t].to(x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, cache, start_pos)
        if last_only:
            x = x[:, -1:]
        return self.lm_head(self.norm(x))

    @torch.inference_mode()
    def forward_batch(self, input_ids: torch.Tensor, batch) -> torch.Tensor:
        """One continuous-batching step.

        input_ids: (N,) new tokens of every scheduled sequence, flattened.
        batch: a PagedBatch with each token's position and the attention layout.
        Returns logits (num_seqs, V) for each sequence's last token only.
        """
        x = self.embed_tokens(input_ids)[None]  # (1, N, hidden)
        cos = self.rope_cos[batch.positions].to(x.dtype)
        sin = self.rope_sin[batch.positions].to(x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, batch, 0)
        return self.lm_head(self.norm(x[0, batch.logits_index]))

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, device, dtype=torch.float16) -> "Transformer":
        model_dir = Path(model_dir)
        cfg = ModelConfig.from_pretrained(model_dir)
        with torch.device("meta"):
            model = cls(cfg)

        state = {}
        for file in sorted(model_dir.glob("*.safetensors")):
            with safe_open(str(file), framework="pt", device="cpu") as f:
                for key in f.keys():
                    state[key.removeprefix("model.")] = f.get_tensor(key).to(dtype)
        if cfg.tie_word_embeddings:
            state.pop("lm_head.weight", None)

        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        if cfg.tie_word_embeddings:
            missing = [k for k in missing if k != "lm_head.weight"]
            model.lm_head.weight = model.embed_tokens.weight
        if missing or unexpected:
            raise RuntimeError(f"weight mismatch: missing={missing} unexpected={unexpected}")

        model = model.to(device).eval()
        model.rope_cos, model.rope_sin = rope_tables(
            cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta, device
        )
        return model
