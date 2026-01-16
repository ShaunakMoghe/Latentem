"""
train_mem_exact.py

Goal:
- Reproduce your "embedding_only = easy, contextual = hard" finding.
- Provide a *learnable* contextual baseline that should actually improve:
    --mode learned_proj

Key idea:
- In contextual hidden-state space, the query vector and the earlier key vector
  are NOT guaranteed to be nearest-neighbors by cosine.
- So we learn small linear projections q_proj/k_proj (+ a value head) to align.

Runs fast: no dataloader, fully synthetic token-id episodes.

Example:
  python -m experiments.train_mem_exact --mode embedding_only --turns 4 --steps 200 --batch 64
  python -m experiments.train_mem_exact --mode contextual     --turns 4 --steps 200 --batch 32
  python -m experiments.train_mem_exact --mode learned_proj   --turns 4 --steps 400 --batch 64 --rep_layer 12
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_safe_ids(tok, n: int, avoid: set[int], rng: random.Random, lo: int = 200, hi: int | None = None) -> List[int]:
    """
    Pick n distinct token IDs avoiding special tokens and anything in `avoid`.
    We also skip very-low IDs which often correspond to special-ish tokens.
    """
    if hi is None:
        hi = tok.vocab_size - 1
    out = []
    tried = 0
    while len(out) < n:
        tried += 1
        if tried > 2_000_000:
            raise RuntimeError("Could not sample enough safe token IDs.")
        tid = rng.randint(lo, hi)
        if tid in avoid:
            continue
        out.append(tid)
        avoid.add(tid)
    return out


@dataclass
class BatchEpisode:
    input_ids: torch.Tensor   # (B, T)
    key_positions: torch.Tensor  # (K,) positions of keys within sequence
    val_positions: torch.Tensor  # (K,) positions of values within sequence
    query_pos: int               # scalar: position of query token within sequence
    y_keypos: torch.Tensor       # (B,) which key index is queried (0..K-1)
    y_val: torch.Tensor          # (B,) value class (0..N_CLASSES-1)


def build_batch(
    tok,
    batch: int,
    turns: int,
    class_ids: List[int],
    rng: random.Random,
) -> BatchEpisode:
    """
    Sequence layout:
      [k1, v1, k2, v2, ..., kT, vT, qk]
    where qk equals one of the earlier ki.

    Labels:
      y_keypos = index i of which ki is queried
      y_val    = class index (0..C-1) of vi at that i
    """
    avoid = set(tok.all_special_ids)
    # keys are sampled per-example, but we can reuse a pool safely
    # (doesn't matter; it's synthetic).
    # We'll sample keys fresh per batch for diversity.

    key_positions = torch.tensor([2 * i for i in range(turns)], dtype=torch.long)
    val_positions = torch.tensor([2 * i + 1 for i in range(turns)], dtype=torch.long)
    query_pos = 2 * turns

    T = 2 * turns + 1
    input_ids = torch.empty((batch, T), dtype=torch.long)

    y_keypos = torch.empty((batch,), dtype=torch.long)
    y_val = torch.empty((batch,), dtype=torch.long)

    for b in range(batch):
        local_avoid = set(avoid)  # keep special tokens avoided
        keys = pick_safe_ids(tok, n=turns, avoid=local_avoid, rng=rng, lo=200)
        vals_cls = [rng.randrange(len(class_ids)) for _ in range(turns)]
        vals = [class_ids[c] for c in vals_cls]

        tgt = rng.randrange(turns)
        qk = keys[tgt]

        seq = []
        for i in range(turns):
            seq.append(keys[i])
            seq.append(vals[i])
        seq.append(qk)

        input_ids[b] = torch.tensor(seq, dtype=torch.long)
        y_keypos[b] = tgt
        y_val[b] = vals_cls[tgt]

    return BatchEpisode(
        input_ids=input_ids,
        key_positions=key_positions,
        val_positions=val_positions,
        query_pos=query_pos,
        y_keypos=y_keypos,
        y_val=y_val,
    )


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: (B, D)
    b: (B, K, D)
    returns: (B, K)
    """
    a_n = F.normalize(a, dim=-1)
    b_n = F.normalize(b, dim=-1)
    return torch.einsum("bd,bkd->bk", a_n, b_n)


# -------------------------
# Learned contextual matcher
# -------------------------

class LearnedMatcher(nn.Module):
    """
    Learns to align query vectors with key vectors in contextual hidden-state space,
    and also predict values from value vectors.

    - q_proj: (D -> dproj)
    - k_proj: (D -> dproj)
    - v_proj: (D -> dproj)
    - v_head: (dproj -> n_classes)
    """
    def __init__(self, d_model: int, dproj: int, n_classes: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, dproj, bias=False)
        self.k_proj = nn.Linear(d_model, dproj, bias=False)
        self.v_proj = nn.Linear(d_model, dproj, bias=False)
        self.v_head = nn.Sequential(
            nn.LayerNorm(dproj),
            nn.Linear(dproj, dproj),
            nn.GELU(),
            nn.Linear(dproj, n_classes),
        )

    def forward(
        self,
        q: torch.Tensor,          # (B, D)
        keys: torch.Tensor,       # (B, K, D)
        vals: torch.Tensor,       # (B, K, D)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          pos_logits: (B, K)
          val_logits_attn: (B, C) using softmax(pos_logits)-weighted values
        """
        qh = self.q_proj(q)                     # (B, dproj)
        kh = self.k_proj(keys)                  # (B, K, dproj)
        vh = self.v_proj(vals)                  # (B, K, dproj)

        scale = 1.0 / math.sqrt(qh.shape[-1])
        pos_logits = torch.einsum("bd,bkd->bk", qh, kh) * scale  # (B, K)

        w = F.softmax(pos_logits, dim=-1)       # (B, K)
        read = torch.einsum("bk,bkd->bd", w, vh)  # (B, dproj)

        val_logits_attn = self.v_head(read)     # (B, C)
        return pos_logits, val_logits_attn

    @torch.no_grad()
    def predict_value_from_argmax(
        self,
        pos_logits: torch.Tensor,   # (B, K)
        vals: torch.Tensor,         # (B, K, D)
    ) -> torch.Tensor:
        """
        Predict values using argmax key position (discrete selection).
        Returns predicted classes: (B,)
        """
        pred_pos = pos_logits.argmax(dim=-1)    # (B,)
        # gather value vector at pred_pos
        B, K, D = vals.shape
        idx = pred_pos.view(B, 1, 1).expand(B, 1, D)
        vsel = vals.gather(1, idx).squeeze(1)   # (B, D)
        vsel = self.v_proj(vsel)               # (B, dproj)
        logits = self.v_head(vsel)             # (B, C)
        return logits.argmax(dim=-1)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--mode", type=str, choices=["embedding_only", "contextual", "learned_proj"], default="contextual")
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)

    # contextual controls
    ap.add_argument("--rep_layer", type=int, default=-1,
                    help="Which hidden-state layer to use for contextual. -1 = final layer. Try 8..16 if final is too hard.")
    ap.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp16")

    # learned controls
    ap.add_argument("--dproj", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos_loss_w", type=float, default=1.0)
    ap.add_argument("--val_loss_w", type=float, default=1.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    args = ap.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print(f"🚀 mode={args.mode} model={args.model} device={device} dtype={args.dtype} turns={args.turns}")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Build 10 class token IDs
    rng = random.Random(args.seed + 123)
    avoid = set(tok.all_special_ids)
    class_ids = pick_safe_ids(tok, n=10, avoid=avoid, rng=rng, lo=200)
    n_classes = len(class_ids)

    # Load model (frozen)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype if dtype != torch.float32 else None,
        device_map=None,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    emb_layer = model.get_input_embeddings()

    matcher = None
    opt = None
    if args.mode == "learned_proj":
        d_model = model.config.hidden_size
        matcher = LearnedMatcher(d_model=d_model, dproj=args.dproj, n_classes=n_classes).to(device)
        opt = torch.optim.AdamW(matcher.parameters(), lr=args.lr)

    # Rolling metrics
    total = 0
    total_keypos_ok = 0
    total_val_ok = 0

    # Run
    for step in range(1, args.steps + 1):
        ep = build_batch(tok, args.batch, args.turns, class_ids, rng=rng)

        input_ids = ep.input_ids.to(device)
        y_keypos = ep.y_keypos.to(device)
        y_val = ep.y_val.to(device)

        # Representations
        if args.mode == "embedding_only":
            # Use input embeddings directly -> perfect exact-match retrieval
            with torch.no_grad():
                x = emb_layer(input_ids)  # (B, T, D)
            q = x[:, ep.query_pos, :]  # (B, D)
            keys = x[:, ep.key_positions, :]  # (B, K, D)
            vals = x[:, ep.val_positions, :]  # (B, K, D)

            sims = cosine_sim(q, keys)  # (B, K)
            pred_pos = sims.argmax(dim=-1)
            keypos_ok = (pred_pos == y_keypos).sum().item()

            # "value prediction" is trivial here: gather value token id and map to class index
            B = input_ids.shape[0]
            gather_pos = (2 * pred_pos + 1).view(B, 1)  # value position
            pred_val_token = input_ids.gather(1, gather_pos).squeeze(1)  # (B,)
            # map token id -> class index
            token_to_class = {tid: i for i, tid in enumerate(class_ids)}
            pred_val = torch.tensor([token_to_class[int(t.item())] for t in pred_val_token], device=device)
            val_ok = (pred_val == y_val).sum().item()

            total += B
            total_keypos_ok += keypos_ok
            total_val_ok += val_ok

            if step in (1, 25, 50, 75, 100, 125, 150, 175, 200) or step == args.steps:
                print(f"[step {step}/{args.steps}] value-acc={100*total_val_ok/total:.2f}% | "
                      f"keypos-acc={100*total_keypos_ok/total:.2f}%")

            continue

        # contextual / learned_proj use model hidden states
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
            hs_all = out.hidden_states  # tuple length = n_layers+1 (embeddings + layers)
            layer_idx = args.rep_layer
            if layer_idx < 0:
                layer_idx = len(hs_all) + layer_idx
            hs = hs_all[layer_idx]  # (B, T, D)

        # Move to fp32 for stable similarity/projection math
        hs_f = hs.float()
        q = hs_f[:, ep.query_pos, :]
        keys = hs_f[:, ep.key_positions, :]
        vals = hs_f[:, ep.val_positions, :]

        if args.mode == "contextual":
            sims = cosine_sim(q, keys)
            pred_pos = sims.argmax(dim=-1)
            keypos_ok = (pred_pos == y_keypos).sum().item()

            # Value: gather value token id at predicted pos and map to class index
            B = input_ids.shape[0]
            gather_pos = (2 * pred_pos + 1).view(B, 1)
            pred_val_token = input_ids.gather(1, gather_pos).squeeze(1)
            token_to_class = {tid: i for i, tid in enumerate(class_ids)}
            pred_val = torch.tensor([token_to_class[int(t.item())] for t in pred_val_token], device=device)
            val_ok = (pred_val == y_val).sum().item()

            total += B
            total_keypos_ok += keypos_ok
            total_val_ok += val_ok

            if step in (1, 25, 50, 75, 100, 125, 150, 175, 200) or step == args.steps:
                avg_best_sim = sims.max(dim=-1).values.mean().item()
                print(f"[step {step}/{args.steps}] value-acc={100*total_val_ok/total:.2f}% | "
                      f"keypos-acc={100*total_keypos_ok/total:.2f}% | avg_best_sim={avg_best_sim:.4f}")
            continue

        # learned_proj
        assert matcher is not None and opt is not None

        pos_logits, val_logits_attn = matcher(q, keys, vals)

        loss_pos = F.cross_entropy(pos_logits, y_keypos)
        loss_val = F.cross_entropy(val_logits_attn, y_val)
        loss = args.pos_loss_w * loss_pos + args.val_loss_w * loss_val

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(matcher.parameters(), args.grad_clip)
        opt.step()

        with torch.no_grad():
            pred_pos = pos_logits.argmax(dim=-1)
            keypos_ok = (pred_pos == y_keypos).sum().item()
            pred_val = matcher.predict_value_from_argmax(pos_logits, vals)
            val_ok = (pred_val == y_val).sum().item()

        total += input_ids.shape[0]
        total_keypos_ok += keypos_ok
        total_val_ok += val_ok

        if step in (1, 25, 50, 75, 100, 125, 150, 175, 200) or step == args.steps:
            avg_best_sim = pos_logits.max(dim=-1).values.mean().item()
            print(f"[step {step}/{args.steps}] loss={loss.item():.4f} "
                  f"| value-acc={100*total_val_ok/total:.2f}% "
                  f"| keypos-acc={100*total_keypos_ok/total:.2f}% "
                  f"| avg_best_logit={avg_best_sim:.4f}")

    print(f"✅ Final value-acc: {100*total_val_ok/total:.2f}% over {total} examples")
    print(f"✅ Final keypos-acc: {100*total_keypos_ok/total:.2f}% over {total} examples")
    if args.mode != "embedding_only":
        print(f"ℹ️  Used rep_layer={args.rep_layer}")


if __name__ == "__main__":
    main()
