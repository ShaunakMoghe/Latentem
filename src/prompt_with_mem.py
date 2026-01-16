# src/prompt_with_mem.py
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


@torch.no_grad()
def _embed_ids(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    emb = model.get_input_embeddings()
    return emb(input_ids)


def build_inputs_with_mem(
    model: nn.Module,
    mem_embeds: torch.Tensor,           # (B,K,D)
    kept_ids: torch.Tensor,             # (B,Tk)
    kept_attn: torch.Tensor,            # (B,Tk) 1/0
    ans_ids: Optional[torch.Tensor] = None,    # (B,Ta)
    ans_attn: Optional[torch.Tensor] = None,   # (B,Ta)
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns (inputs_embeds, attention_mask, labels_or_None)

    If ans_ids provided, labels are -100 except on answer tokens.
    """
    B, K, D = mem_embeds.shape

    kept_emb = _embed_ids(model, kept_ids)  # (B,Tk,D)
    mem_attn = torch.ones((B, K), device=kept_ids.device, dtype=kept_attn.dtype)

    if ans_ids is None:
        inputs_embeds = torch.cat([mem_embeds, kept_emb], dim=1)                # (B,K+Tk,D)
        attention_mask = torch.cat([mem_attn, kept_attn], dim=1)                # (B,K+Tk)
        return inputs_embeds, attention_mask, None

    ans_emb = _embed_ids(model, ans_ids)  # (B,Ta,D)
    if ans_attn is None:
        ans_attn = torch.ones((B, ans_ids.size(1)), device=ans_ids.device, dtype=kept_attn.dtype)

    inputs_embeds = torch.cat([mem_embeds, kept_emb, ans_emb], dim=1)          # (B,K+Tk+Ta,D)
    attention_mask = torch.cat([mem_attn, kept_attn, ans_attn], dim=1)         # (B,K+Tk+Ta)

    labels = torch.full((B, K + kept_ids.size(1) + ans_ids.size(1)),
                        -100, device=kept_ids.device, dtype=torch.long)
    labels[:, K + kept_ids.size(1):] = ans_ids

    return inputs_embeds, attention_mask, labels


def build_prompt_with_mem(
    model: nn.Module,
    mem_embeds: torch.Tensor,           # (B,K,D)
    kept_ids: torch.Tensor,             # (B,Tk)
    kept_attn: torch.Tensor,            # (B,Tk)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inference-only helper: returns (inputs_embeds, attention_mask).
    """
    x, m, _ = build_inputs_with_mem(model, mem_embeds, kept_ids, kept_attn, ans_ids=None, ans_attn=None)
    return x, m
