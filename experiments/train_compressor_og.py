# experiments/train_compressor_v0.py
import argparse
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.latent_mem import MemEncoder


@dataclass
class Batch:
    rem_ids: torch.Tensor
    rem_attn: torch.Tensor
    vis_ids: torch.Tensor
    vis_attn: torch.Tensor
    labels: torch.Tensor  # aligned to vis_ids (NOT including mem tokens)


def pick_dtype(name: str):
    name = name.lower()
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {name} (use fp16|bf16|fp32)")


def make_episode(rng: random.Random, turns: int, keep_last: int):
    """
    Build a toy "long context" key-value log.
    We then REMOVE early turns and force the compressor to store them.

    We query a key from the removed region so memory is required.
    """
    keys = []
    for i in range(1, turns + 1):
        v = rng.randint(0, 9)
        keys.append((f"KEY{i:02d}", str(v)))

    removed_n = max(0, turns - keep_last)
    # Query must come from removed region (if possible); otherwise from anywhere
    if removed_n >= 1:
        q_idx = rng.randint(0, removed_n - 1)
    else:
        q_idx = rng.randint(0, turns - 1)

    q_key, q_val = keys[q_idx]

    removed_lines = []
    visible_lines = []
    for t, (k, v) in enumerate(keys, start=1):
        line = f"Turn {t}: {k} = {v}."
        if t <= removed_n:
            removed_lines.append(line)
        else:
            visible_lines.append(line)

    removed_text = "\n".join(removed_lines) if removed_lines else "(nothing removed)"
    visible_text = "\n".join(visible_lines)

    prompt = (
        f"{visible_text}\n"
        f"Question: What is {q_key}?\n"
        f"Answer:"
    )
    target = f" {q_val}"

    return removed_text, prompt, target


def collate(tok, device, removed_texts, prompts, targets, max_len=256):
    # removed chunk
    rem = tok(
        removed_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    rem_ids = rem["input_ids"].to(device)
    rem_attn = rem["attention_mask"].to(device)

    # tokenize prompt-only to find where answer starts per example
    prompt_enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    prompt_ids = prompt_enc["input_ids"]
    prompt_attn = prompt_enc["attention_mask"]
    prompt_lens = prompt_attn.long().sum(dim=1)  # (B,)

    # full = prompt + target
    full_texts = [p + t for p, t in zip(prompts, targets)]
    vis = tok(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    vis_ids = vis["input_ids"].to(device)
    vis_attn = vis["attention_mask"].to(device)

    labels = vis_ids.clone()
    # mask padding
    labels[vis_attn == 0] = -100
    # mask everything before answer start
    for i in range(labels.size(0)):
        labels[i, : prompt_lens[i].item()] = -100

    return Batch(rem_ids, rem_attn, vis_ids, vis_attn, labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--rep_layer", type=int, default=12)
    ap.add_argument("--k_mem", type=int, default=8)
    ap.add_argument("--turns", type=int, default=14)
    ap.add_argument("--keep_last", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dtype", type=str, default="bf16")  # bf16 is way more stable than fp16
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="logs/mem_encoder_v0.pt")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm_dtype = pick_dtype(args.dtype)

    print(
        f"🚀 Training MemEncoder: model={args.model} device={device} "
        f"dtype={args.dtype} rep_layer={args.rep_layer} K={args.k_mem} "
        f"turns={args.turns} keep_last={args.keep_last}"
    )

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Load frozen LM
    # transformers versions differ: some prefer dtype=..., some torch_dtype=...
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=lm_dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=lm_dtype)

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    d_model = model.config.hidden_size

    # Trainable memory encoder (keep in fp32 for stability)
    mem_enc = MemEncoder(d_model=d_model, K=args.k_mem, n_heads=8).to(device).float()
    opt = torch.optim.AdamW(mem_enc.parameters(), lr=args.lr, weight_decay=1e-4)

    # helper
    embed = model.get_input_embeddings()

    def rep_layer_index(hidden_states):
        L = len(hidden_states)
        return args.rep_layer if args.rep_layer >= 0 else (L + args.rep_layer)

    total_correct = 0
    total_count = 0

    for step in range(1, args.steps + 1):
        removed_texts, prompts, targets = [], [], []
        for _ in range(args.batch):
            rt, pr, tg = make_episode(rng, args.turns, args.keep_last)
            removed_texts.append(rt)
            prompts.append(pr)
            targets.append(tg)

        batch = collate(tok, device, removed_texts, prompts, targets, max_len=args.max_len)

        # 1) get mid-layer reps for REMOVED text under no_grad
        with torch.no_grad():
            out = model(
                input_ids=batch.rem_ids,
                attention_mask=batch.rem_attn,
                output_hidden_states=True,
                use_cache=False,
            )
            hs = out.hidden_states[rep_layer_index(out.hidden_states)]  # (B,T,D)

        # 2) build mem tokens (trainable) in fp32
        mem = mem_enc(hs, batch.rem_attn)  # (B,K,D) fp32
        if not torch.isfinite(mem).all():
            print("⚠️ mem has NaN/Inf — skipping step")
            opt.zero_grad(set_to_none=True)
            continue

        # 3) cast mem into LM dtype before feeding as prefix embeddings
        mem = mem.to(dtype=lm_dtype)

        # 4) embed visible ids
        vis_embeds = embed(batch.vis_ids).to(dtype=lm_dtype)

        # 5) concat prefix + visible
        inputs_embeds = torch.cat([mem, vis_embeds], dim=1)  # (B, K+Tv, D)

        # attention mask
        B = batch.vis_ids.size(0)
        prefix_attn = torch.ones((B, args.k_mem), device=device, dtype=batch.vis_attn.dtype)
        attn = torch.cat([prefix_attn, batch.vis_attn], dim=1)

        # labels (mask prefix positions)
        prefix_labels = torch.full((B, args.k_mem), -100, device=device, dtype=torch.long)
        labels = torch.cat([prefix_labels, batch.labels], dim=1)

        # 6) LM forward (frozen), backprop to mem_enc through inputs_embeds path
        out2 = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            labels=labels,
            use_cache=False,
        )
        loss = out2.loss

        if not torch.isfinite(loss):
            print(f"⚠️ loss is NaN/Inf at step {step} — lowering LR might help. Skipping update.")
            opt.zero_grad(set_to_none=True)
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()

        # grad clip (prevents the fp16 backprop path from nuking you)
        torch.nn.utils.clip_grad_norm_(mem_enc.parameters(), args.grad_clip)

        opt.step()

        # quick accuracy: did it predict the correct answer token (first answer token)
        # We'll compute token-level correctness on the FIRST unmasked label position per example.
        with torch.no_grad():
            logits = out2.logits  # (B, K+Tv, V)
            preds = logits.argmax(dim=-1)

            # find first label != -100 per row
            for i in range(B):
                row = labels[i]
                idxs = (row != -100).nonzero(as_tuple=False)
                if idxs.numel() == 0:
                    continue
                j = idxs[0].item()
                total_count += 1
                total_correct += int(preds[i, j - 1].item() == row[j].item())
                # note: LM predicts token j using logits at position j-1 (causal shift)

        if step % args.log_every == 0 or step == 1:
            acc = (total_correct / max(1, total_count)) * 100.0
            print(f"[step {step}/{args.steps}] loss={loss.item():.4f} answer-acc={acc:.2f}%")

    # save
    torch.save(mem_enc.state_dict(), args.save)
    print(f"✅ saved to {args.save}")


if __name__ == "__main__":
    main()
