"""
debug_contextual_internals.py

Probe why contextual matching fails:
- Computes per-layer cosine retrieval accuracy for keys/values.
- Shows how query-key similarity behaves by position.
- Optional: --attn to print last-layer attention from query to earlier tokens.

Example:
  python -m experiments.debug_contextual_internals --turns 4 --steps 200 --batch 32
  python -m experiments.debug_contextual_internals --turns 4 --steps 200 --batch 16 --attn
"""

from __future__ import annotations

import argparse
import math
import random
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_safe_ids(tok, n: int, avoid: set[int], rng: random.Random, lo: int = 200) -> List[int]:
    out = []
    while len(out) < n:
        tid = rng.randint(lo, tok.vocab_size - 1)
        if tid in avoid:
            continue
        avoid.add(tid)
        out.append(tid)
    return out


def build_batch(tok, batch: int, turns: int, class_ids: List[int], rng: random.Random):
    # same layout as train_mem_exact:
    # [k1 v1 k2 v2 ... kT vT qk]
    key_positions = torch.tensor([2 * i for i in range(turns)], dtype=torch.long)
    val_positions = torch.tensor([2 * i + 1 for i in range(turns)], dtype=torch.long)
    query_pos = 2 * turns
    T = 2 * turns + 1

    input_ids = torch.empty((batch, T), dtype=torch.long)
    y_keypos = torch.empty((batch,), dtype=torch.long)
    y_val = torch.empty((batch,), dtype=torch.long)

    base_avoid = set(tok.all_special_ids)

    for b in range(batch):
        avoid = set(base_avoid)
        keys = pick_safe_ids(tok, turns, avoid, rng)
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

    return input_ids, y_keypos, y_val, key_positions, val_positions, query_pos


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return torch.einsum("bd,bkd->bk", a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--attn", action="store_true", help="Attempt to output attentions (forces eager if possible).")
    args = ap.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print(f"🔬 Probing contextual internals: model={args.model} device={device} dtype={args.dtype} turns={args.turns}")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    rng = random.Random(args.seed + 999)
    avoid = set(tok.all_special_ids)
    class_ids = pick_safe_ids(tok, 10, avoid, rng)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype if dtype != torch.float32 else None,
        device_map=None,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # If you requested attentions, try to force eager attention backend if possible.
    if args.attn:
        # Some transformers versions support this method.
        if hasattr(model, "set_attn_implementation"):
            try:
                model.set_attn_implementation("eager")
            except Exception:
                pass

    # Accumulators per-layer
    # layers: 0 = embeddings, 1..n = transformer blocks, last index = final hidden
    per_layer_key_ok = None
    per_layer_val_ok = None
    per_layer_true_sim = None
    per_layer_best_sim = None
    per_layer_margin = None
    total = 0

    # A few live prints
    report_steps = {1, 25, 50, 100, args.steps}

    for step in range(1, args.steps + 1):
        input_ids, y_keypos, y_val, key_pos, val_pos, query_pos = build_batch(
            tok, args.batch, args.turns, class_ids, rng
        )
        input_ids = input_ids.to(device)
        y_keypos = y_keypos.to(device)
        y_val = y_val.to(device)

        out = model(
            input_ids=input_ids,
            output_hidden_states=True,
            output_attentions=bool(args.attn),
            use_cache=False,
        )
        hs_all = out.hidden_states  # tuple

        if per_layer_key_ok is None:
            L = len(hs_all)
            per_layer_key_ok = torch.zeros(L, device="cpu", dtype=torch.long)
            per_layer_val_ok = torch.zeros(L, device="cpu", dtype=torch.long)
            per_layer_true_sim = torch.zeros(L, device="cpu", dtype=torch.float)
            per_layer_best_sim = torch.zeros(L, device="cpu", dtype=torch.float)
            per_layer_margin = torch.zeros(L, device="cpu", dtype=torch.float)

        B = input_ids.shape[0]
        total += B

        # Evaluate each layer
        for li, hs in enumerate(hs_all):
            hs = hs.float()

            q = hs[:, query_pos, :]               # (B, D)
            keys = hs[:, key_pos, :]              # (B, K, D)
            vals = hs[:, val_pos, :]              # (B, K, D)

            sims = cosine_sim(q, keys)            # (B, K)
            best_sim, best_idx = sims.max(dim=-1) # (B,)
            true_sim = sims.gather(1, y_keypos.view(B, 1)).squeeze(1)
            margin = (best_sim - true_sim)

            pred_pos = best_idx
            key_ok = (pred_pos == y_keypos).sum().item()

            # value correctness from predicted position: token-id mapping
            gather_val_pos = (2 * pred_pos + 1).view(B, 1)
            pred_val_token = input_ids.gather(1, gather_val_pos).squeeze(1)
            token_to_class = {tid: i for i, tid in enumerate(class_ids)}
            pred_val = torch.tensor([token_to_class[int(t.item())] for t in pred_val_token], device=device)
            val_ok = (pred_val == y_val).sum().item()

            per_layer_key_ok[li] += key_ok
            per_layer_val_ok[li] += val_ok
            per_layer_true_sim[li] += true_sim.mean().item()
            per_layer_best_sim[li] += best_sim.mean().item()
            per_layer_margin[li] += margin.mean().item()

        # Live print on final layer
        if step in report_steps:
            li = len(hs_all) - 1
            key_acc = 100.0 * (per_layer_key_ok[li].item() / total)
            val_acc = 100.0 * (per_layer_val_ok[li].item() / total)
            avg_true = (per_layer_true_sim[li].item() / step)
            avg_best = (per_layer_best_sim[li].item() / step)
            avg_margin = (per_layer_margin[li].item() / step)
            print(f"[step {step}/{args.steps}] last-layer key-acc={key_acc:.2f}% val-acc={val_acc:.2f}% "
                  f"avg_true_sim={avg_true:.4f} avg_best_sim={avg_best:.4f} avg_margin={avg_margin:.4f}")

    # Final per-layer table
    print("\n===== Per-layer summary (0=embeddings, last=final hidden) =====")
    L = len(per_layer_key_ok)
    for li in range(L):
        key_acc = 100.0 * per_layer_key_ok[li].item() / total
        val_acc = 100.0 * per_layer_val_ok[li].item() / total
        avg_true = per_layer_true_sim[li].item() / args.steps
        avg_best = per_layer_best_sim[li].item() / args.steps
        avg_margin = per_layer_margin[li].item() / args.steps
        print(f"layer {li:2d}: key-acc={key_acc:6.2f}% | val-acc={val_acc:6.2f}% | "
              f"avg_true_sim={avg_true:.4f} | avg_best_sim={avg_best:.4f} | avg_margin={avg_margin:.4f}")

    # Optional attention diagnostics (if present)
    if args.attn:
        atts = out.attentions
        if atts is None:
            print("\n⚠️  output_attentions was requested but returned None. (Backend likely not eager.)")
        else:
            # last layer attention: typically (B, H, T, T)
            last = atts[-1]
            if last is None:
                print("\n⚠️  Last attention tensor is None.")
            else:
                last = last.float()
                # attention FROM query token to previous positions 0..T-1
                qpos = (2 * args.turns)
                attn_q = last[:, :, qpos, :qpos]  # (B, H, qpos)
                attn_avg = attn_q.mean(dim=(0, 1))  # (qpos,)
                print("\n===== Last-layer query attention (avg over batch & heads) =====")
                print("attn to prev positions 0..T-1:", [round(float(x), 4) for x in attn_avg.tolist()])


if __name__ == "__main__":
    main()
