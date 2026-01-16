"""
Train a conditional latent-memory encoder (digit-first task) WITH LoRA ADAPTATION.

Goal: compress removed context representations into K "memory slots".
CHANGE: We now use LoRA on the base LLM so it can learn to READ these slots.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.latent_mem import MemEncoder
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict  # <--- NEW IMPORT


# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_flat(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = a.flatten(1)
    b = b.flatten(1)
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a * b).sum(dim=-1)


def pairwise_cos_stats(x: torch.Tensor, max_pairs: int = 64) -> Tuple[float, float, float]:
    """Cosine stats over random pairs in batch. x: (B, ...)"""
    b = x.shape[0]
    if b < 2:
        return 1.0, 1.0, 1.0
    idx = torch.randperm(b, device=x.device)
    pairs = []
    for i in range(0, min(b - 1, 2 * max_pairs), 2):
        pairs.append((idx[i].item(), idx[i + 1].item()))
    a = torch.stack([x[i].flatten() for i, _ in pairs], dim=0)
    c = torch.stack([x[j].flatten() for _, j in pairs], dim=0)
    cs = cosine_flat(a, c).detach().float().cpu()
    return float(cs.min()), float(cs.mean()), float(cs.max())


# -------------------------
# Data
# -------------------------

KEYS = [f"KEY{i:02d}" for i in range(1, 11)]
DIGITS = list(range(10))


@dataclass
class Episode:
    removed_text: str
    kept_text: str
    target_key: str
    target_val: int


def make_episode(rng: random.Random, turns: int, keep_last: int) -> Episode:
    assert turns > keep_last, "need some removed turns to actually test memory"

    kv = [(rng.choice(KEYS), rng.choice(DIGITS)) for _ in range(turns)]
    removed = kv[: turns - keep_last]
    kept = kv[turns - keep_last :]

    # Force target in removed
    target_key, target_val = rng.choice(removed)

    removed_lines = [f"{v}" for t, (k, v) in enumerate(removed)]
    kept_lines = [f"Turn {t+len(removed)}: {k} = {v}." for t, (k, v) in enumerate(kept)]

    removed_text = "\n".join(removed_lines)
    kept_text = "\n".join(kept_lines) + f"\nQuestion: what is the value of {target_key}?\nAnswer:"

    return Episode(
        removed_text=removed_text,
        kept_text=kept_text,
        target_key=target_key,
        target_val=target_val,
    )


def tokenize_batch(tok, texts: List[str], max_len: int, device: torch.device):
    out = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    )
    return {k: v.to(device) for k, v in out.items()}


# -------------------------
# Oracle / anti-collapse losses
# -------------------------

def make_oracle_mem(
    tok,
    embed_layer: torch.nn.Module,
    target_key: str,
    target_val: int,
    k_mem: int,
    device: torch.device,
    kind: str = "gold_kv",
) -> torch.Tensor:
    if kind == "gold_digit":
        s = f"{target_val}"
    else:
        s = f"{target_key} = {target_val}."
    ids = tok(s, add_special_tokens=False)["input_ids"]
    ids_t = torch.tensor(ids, device=device, dtype=torch.long)  # (T,)
    embs = embed_layer(ids_t)  # (T,D)

    # Chunk into K slots (contiguous chunks), mean-pool per chunk.
    t = embs.shape[0]
    chunk = math.ceil(t / k_mem)
    slots = []
    for i in range(k_mem):
        a = i * chunk
        b = min((i + 1) * chunk, t)
        if a >= b:
            slots.append(torch.zeros((embs.shape[1],), device=device, dtype=embs.dtype))
        else:
            slots.append(embs[a:b].mean(dim=0))
    return torch.stack(slots, dim=0)  # (K,D)


def oracle_alignment_loss(mem: torch.Tensor, oracle: torch.Tensor) -> torch.Tensor:
    mem_u = F.normalize(mem.float(), dim=-1)
    oracle_u = F.normalize(oracle.float(), dim=-1)
    return 1.0 - (mem_u * oracle_u).sum(dim=-1).mean()


def diversity_loss(mem: torch.Tensor) -> torch.Tensor:
    pooled = mem.mean(dim=1)  # (B,D)
    pooled = F.normalize(pooled.float(), dim=-1)
    sim = pooled @ pooled.t()  # (B,B)
    b = sim.shape[0]
    eye = torch.eye(b, device=sim.device, dtype=torch.bool)
    off = sim[~eye]
    return (off ** 2).mean()


# -------------------------
# Training / Debug
# -------------------------

@torch.no_grad()
def encode_reps(model, input_ids, attn_mask, rep_layer: int) -> torch.Tensor:
    # IMPORTANT: We need the BASE model's hidden states, not the LoRA-adapted ones 
    # for the input to the compressor. 
    # Fortunately, PEFT wraps the base model. calling model(...) runs LoRA forward.
    # To get "raw" hidden states, we can usually just run it. The compressor will 
    # learn to adapt to whatever the LoRA-modded model outputs at layer X.
    out = model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hs = out.hidden_states
    if rep_layer < 0:
        rep_layer = len(hs) + rep_layer
    return hs[rep_layer]  # (B,T,D)


def build_batch(
    tok,
    episodes: List[Episode],
    max_len: int,
    device: torch.device,
):
    removed_texts = [e.removed_text for e in episodes]
    rem = tokenize_batch(tok, removed_texts, max_len=max_len, device=device)

    keep_full_ids_list = []
    labels_list = []

    for e in episodes:
        prompt_ids = tok(e.kept_text, add_special_tokens=False)["input_ids"]
        full_ids = tok(e.kept_text + str(e.target_val), add_special_tokens=False)["input_ids"]

        if len(full_ids) > max_len:
            full_ids = full_ids[-max_len:]
            prompt_ids = tok(e.kept_text, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]

        added = full_ids[len(prompt_ids):]
        if len(added) == 0:
            added = tok(str(e.target_val), add_special_tokens=False)["input_ids"]

        lab = [-100] * (len(full_ids) - len(added)) + added

        keep_full_ids_list.append(torch.tensor(full_ids, device=device, dtype=torch.long))
        labels_list.append(torch.tensor(lab, device=device, dtype=torch.long))

    keep_full_ids = torch.nn.utils.rnn.pad_sequence(
        keep_full_ids_list, batch_first=True, padding_value=tok.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)

    keep_full_attn = (keep_full_ids != tok.pad_token_id).long()
    return rem, keep_full_ids, keep_full_attn, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--rep_layer", type=int, default=12)
    ap.add_argument("--k_mem", type=int, default=8)
    ap.add_argument("--turns", type=int, default=14)
    ap.add_argument("--keep_last", type=int, default=4)

    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--save", type=str, default="logs/mem_encoder_v1_digitfirst.pt")

    ap.add_argument("--no_mem_norm", action="store_true", help="disable mem_norm scaling")
    ap.add_argument("--oracle_alpha", type=float, default=1.0, help="aux weight for oracle alignment loss")
    ap.add_argument("--oracle_type", type=str, default="gold_kv", choices=["none", "gold_kv", "gold_digit"])
    ap.add_argument("--diversity_alpha", type=float, default=0.0)

    ap.add_argument("--debug_cond_only", action="store_true")
    ap.add_argument("--overfit_one", action="store_true")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 1. Load Base Model (Do NOT freeze manually here)
    print(f"Loading model {args.model}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)

    # 2. Apply LoRA (This adapts the attention heads to read latent memory)
    print("Applying LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dim = model.get_input_embeddings().weight.shape[1]
    mem_enc = MemEncoder(dim=dim, k_mem=args.k_mem, mem_norm=not args.no_mem_norm).to(device=device)
    mem_enc.train()
    mem_enc = mem_enc.float()  # keep mem encoder fp32 for stability

    # 3. Optimize BOTH the Encoder AND the LoRA adapters
    opt = torch.optim.AdamW([
        {'params': mem_enc.parameters()},
        {'params': model.parameters()}
    ], lr=args.lr)

    rng = random.Random(args.seed)

    print(
        f"🚀 Training MemEncoder+LoRA (digit-first): model={args.model} device={device} dtype={args.dtype} "
        f"rep_layer={args.rep_layer} K={args.k_mem} turns={args.turns} keep_last={args.keep_last} mem_norm={not args.no_mem_norm} "
        f"oracle={args.oracle_type} oracle_alpha={args.oracle_alpha} div_alpha={args.diversity_alpha}"
    )

    fixed_ep: Optional[Episode] = None
    if args.overfit_one:
        fixed_ep = make_episode(rng, turns=args.turns, keep_last=args.keep_last)
        print("[train-dbg] overfit_one enabled: repeating the same episode every batch")

    # Access base model embeddings via PEFT wrapper
    embed_layer = model.get_input_embeddings()

    for step in range(1, args.steps + 1):
        episodes = []
        for _ in range(args.batch):
            episodes.append(fixed_ep if fixed_ep is not None else make_episode(rng, args.turns, args.keep_last))

        rem, keep_full_ids, keep_full_attn, labels = build_batch(tok=tok, episodes=episodes, max_len=args.max_len, device=device)

        with torch.no_grad():
            reps = encode_reps(model, rem["input_ids"], rem["attention_mask"], rep_layer=args.rep_layer)

        mem = mem_enc(
            reps.float(),
            reps_attn=rem["attention_mask"],
            visible_embeds=embed_layer(keep_full_ids).float() if (not args.no_mem_norm) else None,
            visible_attn=keep_full_attn if (not args.no_mem_norm) else None,
        )  # (B,K,D) fp32

        if args.debug_cond_only:
            rem_kept = rem["attention_mask"].float().sum(dim=1)
            rem_pad = (rem["attention_mask"] == 0).float().sum(dim=1)
            print(
                f"[train-dbg] rem_attn kept tokens: min={rem_kept.min().item():.1f} mean={rem_kept.mean().item():.1f} max={rem_kept.max().item():.1f} | "
                f"pad tokens: min={rem_pad.min().item():.1f} mean={rem_pad.mean().item():.1f} max={rem_pad.max().item():.1f}"
            )
            mn, me, mx = pairwise_cos_stats(mem)
            print(f"[train-dbg] conditioning cos(mem[i],mem[j]) over batch pairs: min={mn:.4f} mean={me:.4f} max={mx:.4f}")
            print(f"[train-dbg] mem mean L2 norm (pre-cast): {mem.norm(dim=-1).mean().item():.4f}")
            return

        keep_emb = embed_layer(keep_full_ids)  # LM dtype
        mem_cast = mem.to(dtype=keep_emb.dtype)
        prefix = torch.cat([mem_cast, keep_emb], dim=1)  # (B,K+S,D)

        attn = torch.cat(
            [torch.ones((keep_full_attn.shape[0], args.k_mem), device=device, dtype=keep_full_attn.dtype), keep_full_attn],
            dim=1,
        )
        full_labels = torch.cat(
            [torch.full((labels.shape[0], args.k_mem), -100, device=device, dtype=labels.dtype), labels],
            dim=1,
        )

        # Forward pass (trains LoRA weights)
        out = model(inputs_embeds=prefix, attention_mask=attn, labels=full_labels, use_cache=False)
        loss_lm = out.loss

        loss_oracle = torch.tensor(0.0, device=device)
        if args.oracle_type != "none" and args.oracle_alpha > 0:
            oracle_slots = []
            for e in episodes:
                oracle_slots.append(
                    make_oracle_mem(
                        tok,
                        embed_layer,
                        e.target_key,
                        e.target_val,
                        args.k_mem,
                        device,
                        kind=args.oracle_type,
                    )
                )
            oracle = torch.stack(oracle_slots, dim=0)  # (B,K,D)
            loss_oracle = oracle_alignment_loss(mem, oracle)

        loss_div = torch.tensor(0.0, device=device)
        if args.diversity_alpha > 0:
            loss_div = diversity_loss(mem)

        loss = loss_lm + args.oracle_alpha * loss_oracle + args.diversity_alpha * loss_div

        opt.zero_grad(set_to_none=True)
        loss.backward()

        if args.grad_clip is not None and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(mem_enc.parameters(), args.grad_clip)

        opt.step()

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                logits = out.logits
                sup = (full_labels != -100)
                if sup.any():
                    idx = sup.long().sum(dim=1) - 1
                    gathered = logits[torch.arange(logits.size(0), device=device), idx, :]
                    pred = gathered.argmax(dim=-1)
                    gold = full_labels[torch.arange(full_labels.size(0), device=device), idx]
                    acc = (pred == gold).float().mean().item() * 100.0
                else:
                    acc = 0.0
            print(
                f"[step {step}/{args.steps}] loss={loss.item():.4f} lm={loss_lm.item():.4f} "
                f"oracle={float(loss_oracle.item()):.4f} div={float(loss_div.item()):.4f} answer-acc={acc:.2f}%"
            )

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    # Save State Dictionary with LoRA weights
    ckpt = {
        "state_dict": mem_enc.state_dict(),
        "lora_state_dict": get_peft_model_state_dict(model), # <--- NEW: Save adapter weights
        "meta": {
            "model": args.model,
            "rep_layer": args.rep_layer,
            "k_mem": args.k_mem,
            "turns": args.turns,
            "keep_last": args.keep_last,
            "mem_norm": (not args.no_mem_norm),
            "oracle_type": args.oracle_type,
            "oracle_alpha": args.oracle_alpha,
            "dtype": args.dtype,
        },
    }
    torch.save(ckpt, args.save)
    print(f"✅ saved to {args.save}")


if __name__ == "__main__":
    main()