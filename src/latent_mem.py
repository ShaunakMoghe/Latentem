"""
src/latent_mem.py

Learned latent memory compressor (MemEncoder) used by training/eval scripts.

Key fixes vs earlier versions:
- Robust dtype handling: avoid bf16-vs-fp32 matmul errors by casting inputs to param dtype.
- Anti-collapse conditioning on repetitive templates: use learned attention pooling (AttnPool)
  instead of pure mean pooling so the conditioner can focus on informative tokens (digits).
- Backward-compatible MemEncoder init/forward signatures:
    MemEncoder(dim=..., k_mem=..., ...) or MemEncoder(d_model=..., K=..., ...)
    forward(reps, attn_mask=...) or forward(reps, reps_attn=..., visible_embeds=..., visible_attn=...)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Mask helpers
# -------------------------

def attn_mask_to_key_padding(attn_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Convert an attention mask (1=keep, 0=pad) into key_padding_mask (True=pad) for MHA.
    Accepts bool or int/float masks.
    """
    if attn_mask is None:
        return None
    if attn_mask.dtype == torch.bool:
        keep = attn_mask
    else:
        keep = attn_mask != 0
    return ~keep  # True where pad


def masked_mean(x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    x: (B,T,D)
    attn_mask: (B,T) 1=keep, 0=pad (or bool)
    returns: (B,D)
    """
    if attn_mask is None:
        return x.mean(dim=1)
    if attn_mask.dtype == torch.bool:
        keep = attn_mask
    else:
        keep = attn_mask != 0
    w = keep.to(dtype=x.dtype).unsqueeze(-1)  # (B,T,1)
    denom = w.sum(dim=1).clamp_min(1.0)
    return (x * w).sum(dim=1) / denom


def masked_mean_token_norm(x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    x: (B,T,D) token embeddings (or reps)
    attn_mask: (B,T)
    returns: (B,1,1) mean L2 token norm, used to scale mem slots when mem_norm is enabled
    """
    tok_norm = x.norm(dim=-1)  # (B,T)
    if attn_mask is None:
        m = tok_norm.mean(dim=1)
    else:
        if attn_mask.dtype == torch.bool:
            keep = attn_mask
        else:
            keep = attn_mask != 0
        w = keep.to(dtype=tok_norm.dtype)
        denom = w.sum(dim=1).clamp_min(1.0)
        m = (tok_norm * w).sum(dim=1) / denom
    return m.view(-1, 1, 1)


# -------------------------
# Norm + projector
# -------------------------

class RMSNorm(nn.Module):
    """
    Actually LayerNorm now, but kept name to avoid refactoring everything.
    LayerNorm centers the mean, preventing 'mean drift' collapse.
    """
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(d, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)


class RepProjector(nn.Module):
    """
    Small residual MLP projector that stays stable and keeps the original signal.
    """
    def __init__(self, d: int, hidden_mult: int = 2):
        super().__init__()
        h = hidden_mult * d
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)
        self.act = nn.GELU()
        self.norm = RMSNorm(d)

        # init
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure dtype matches weights to avoid "mat1/mat2 dtype mismatch"
        w_dtype = self.fc1.weight.dtype
        if x.dtype != w_dtype:
            x = x.to(dtype=w_dtype)

        y = self.fc2(self.act(self.fc1(x)))
        # residual keeps base-model reps intact while allowing gentle adaptation
        return self.norm(x + 0.5 * y)


# -------------------------
# Attention pooling conditioner
# -------------------------

class AttnPool(nn.Module):
    """
    Learnable attention pooling over tokens:
      pooled = sum_t softmax(<x_t, pool_query>) * x_t
    This avoids mean-pooling washing out rare informative tokens on repetitive templates.
    """
    def __init__(self, d: int):
        super().__init__()
        self.pool_query = nn.Parameter(torch.randn(d) * 0.02)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # x: (B,T,D)
        B, T, D = x.shape
        scores = (x * self.pool_query.view(1, 1, D)).sum(dim=-1) / math.sqrt(D)  # (B,T)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                keep = attn_mask
            else:
                keep = attn_mask != 0
            scores = scores.masked_fill(~keep, -1e9)

        w = F.softmax(scores, dim=1)  # (B,T)
        w = w.to(dtype=x.dtype).unsqueeze(-1)  # (B,T,1)

        pooled = (x * w).sum(dim=1)  # (B,D)

        # Safety: if a row was fully masked, softmax can produce NaNs.
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        return pooled


# -------------------------
# Latent compressor
# -------------------------

class LatentCompressor(nn.Module):
    """
    Cross-attention from K learned memory queries into the removed-token reps.
    Conditioning shift is derived from BOTH:
      - masked mean pooling (stable baseline)
      - learned attention pooling (lets the conditioner focus on digits)
    """
    def __init__(self, d_model: int, K: int = 8, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.K = int(K)
        self.d_model = int(d_model)

        self.mem_queries = nn.Parameter(torch.randn(self.K, self.d_model) * 0.02)

        # conditioner
        self.pool = AttnPool(self.d_model)
        self.q_from_pooled = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            # nn.Tanh(), # REMOVED to allow stronger shift
        )
        for m in self.q_from_pooled:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.2) # BOOSTED from 0.02
                nn.init.zeros_(m.bias)

        self.q_scale = nn.Parameter(torch.tensor(1.0))
        self.shift_dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

        self.q_norm = RMSNorm(self.d_model)
        self.kv_norm = RMSNorm(self.d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ff = nn.Sequential(
            nn.Linear(self.d_model, 4 * self.d_model),
            nn.GELU(),
            nn.Linear(4 * self.d_model, self.d_model),
        )
        for m in self.ff:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                nn.init.zeros_(m.bias)

        self.out_norm = RMSNorm(self.d_model)

    def forward(self, x: torch.Tensor, seq_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B,T,D) removed reps (already projected)
        seq_attn_mask: (B,T) 1=keep, 0=pad
        returns: mem (B,K,D)
        """
        B, T, D = x.shape

        # Base queries (shared) + per-example conditioner shift
        q = self.mem_queries.unsqueeze(0).expand(B, -1, -1)  # (B,K,D)

        pooled_mean = masked_mean(x, seq_attn_mask)          # (B,D)
        pooled_attn = self.pool(x, seq_attn_mask)            # (B,D)
        pooled = pooled_mean + pooled_attn                   # (B,D)

        shift = self.q_scale * self.q_from_pooled(pooled)    # (B,D)
        
        # DEBUG VARIANCE (ALWAYS)
        # if shift.requires_grad:
        print(f"    [Dg internal] Shift Var:    {shift.detach().float().var(dim=0).mean().item():.6f}")
             
        shift = self.shift_dropout(shift)
        q = q + shift.unsqueeze(1)                           # (B,K,D)
        
        print(f"    [Dg internal] Q Var:        {q.detach().float().var(dim=0).mean().item():.6f}")

        q = self.q_norm(q)
        kv = self.kv_norm(x)
        
        key_padding_mask = attn_mask_to_key_padding(seq_attn_mask)
        
        # Attention in fp32 for stability; cast back afterwards
        attn_out, _ = self.attn(
            query=q.float(),
            key=kv.float(),
            value=kv.float(),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attn_out = attn_out.to(dtype=q.dtype)
        print(f"    [Dg internal] AttnOut Var:  {attn_out.detach().float().var(dim=0).mean().item():.6f}")
        
        h = q + attn_out
        # h = h + self.ff(h) # DISABLE FF 
        print(f"    [Dg internal] H PreNorm Var:{h.detach().float().var(dim=0).mean().item():.6f}")
        
        h = self.out_norm(h)
        
        # if h.requires_grad:
        print(f"    [Dg internal] Output H Var: {h.detach().float().var(dim=0).mean().item():.6f}")

        return h


# -------------------------
# MemEncoder (public)
# -------------------------

class MemEncoder(nn.Module):
    """
    Backward-compatible MemEncoder.

    Init supports:
      - MemEncoder(dim=..., k_mem=..., n_heads=8, mem_norm=True)
      - MemEncoder(d_model=..., K=..., n_heads=8, mem_norm=True)

    Forward supports:
      - forward(reps, attn_mask=...)
      - forward(reps, reps_attn=...)
      - forward(..., visible_embeds=..., visible_attn=...) for norm matching when mem_norm is enabled
    """
    def __init__(
        self,
        dim: Optional[int] = None,
        k_mem: Optional[int] = None,
        *,
        d_model: Optional[int] = None,
        K: Optional[int] = None,
        n_heads: int = 8,
        mem_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        if d_model is None:
            d_model = dim
        if K is None:
            K = k_mem
        if d_model is None or K is None:
            raise TypeError(
                "MemEncoder requires (dim, k_mem) or (d_model, K). "
                f"Got dim={dim}, k_mem={k_mem}, d_model={d_model}, K={K}."
            )

        self.d_model = int(d_model)
        self.K = int(K)
        self.mem_norm = bool(mem_norm)

        self.rep_proj = RepProjector(self.d_model)
        self.comp = LatentCompressor(d_model=self.d_model, K=self.K, n_heads=n_heads, dropout=dropout)

    def forward(
        self,
        reps: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        *,
        reps_attn: Optional[torch.Tensor] = None,
        visible_embeds: Optional[torch.Tensor] = None,
        visible_attn: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        reps: (B,T,D) removed segment base-model hidden states
        attn_mask / reps_attn: (B,T) where 1=keep, 0=pad
        visible_embeds: (B,S,D) kept token embeddings (optional; used to scale mem slots)
        visible_attn: (B,S) attention mask for visible_embeds (optional)
        returns: mem (B,K,D)
        """
        seq_mask = reps_attn if reps_attn is not None else attn_mask

        # Force reps to module dtype to prevent bf16/float matmul errors
        p_dtype = next(self.parameters()).dtype
        if reps.dtype != p_dtype:
            reps = reps.to(dtype=p_dtype)

        x = self.rep_proj(reps)
        mem = self.comp(x, seq_mask)

        if self.mem_norm:
            mem_unit = mem / (mem.norm(dim=-1, keepdim=True) + 1e-8)

            if visible_embeds is not None:
                # also cast visible_embeds to param dtype for consistent norm stats
                if visible_embeds.dtype != p_dtype:
                    visible_embeds = visible_embeds.to(dtype=p_dtype)
                scale = masked_mean_token_norm(visible_embeds, visible_attn)  # (B,1,1)
                mem = mem_unit * scale
            else:
                mem = mem_unit

        return mem
