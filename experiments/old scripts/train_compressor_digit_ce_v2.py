# experiments/train_compressor_digit_ce_v2.py
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.latent_mem import MemEncoder


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
    keys = []
    for i in range(1, turns + 1):
        v = rng.randint(0, 9)
        keys.append((f"KEY{i:02d}", str(v)))

    removed_n = max(0, turns - keep_last)
    if removed_n >= 1:
        q_idx = rng.randint(0, removed_n - 1)  # force target in removed region (matches your eval flag)
    else:
        q_idx = rng.randint(0, turns - 1)

    q_key, q_val = keys[q_idx]

    removed_lines, visible_lines = [], []
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

    return removed_text, prompt, int(q_val)


@dataclass
class Batch:
    rem_ids: torch.Tensor
    rem_attn: torch.Tensor
    prompt_ids: torch.Tensor
    prompt_attn: torch.Tensor
    prompt_lens: torch.Tensor
    gold_digits: torch.Tensor  # (B,)


def collate(tok, device, removed_texts, prompts, gold_digits, max_len=256) -> Batch:
    rem = tok(
        removed_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    prompt_enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )

    rem_ids = rem["input_ids"].to(device)
    rem_attn = rem["attention_mask"].to(device)

    prompt_ids = prompt_enc["input_ids"].to(device)
    prompt_attn = prompt_enc["attention_mask"].to(device)
    prompt_lens = prompt_attn.long().sum(dim=1)  # (B,)

    gold_digits = torch.tensor(gold_digits, device=device, dtype=torch.long)
    return Batch(rem_ids, rem_attn, prompt_ids, prompt_attn, prompt_lens, gold_digits)


def get_digit_candidates(tok) -> Tuple[List[List[int]], List[int], List[int]]:
    """
    Candidate continuations: " {d}" with add_special_tokens=False.
    Returns:
      cand_ids_list[d]  : token ids list
      cand_last_ids[d]  : last token id (digit token)
      cand_lens[d]      : length of token id list
    """
    cand_ids_list = []
    cand_last_ids = []
    cand_lens = []
    for d in range(10):
        ids = tok(" " + str(d), add_special_tokens=False).input_ids
        cand_ids_list.append(ids)
        cand_last_ids.append(ids[-1])
        cand_lens.append(len(ids))
    return cand_ids_list, cand_last_ids, cand_lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--rep_layer", type=int, default=12)
    ap.add_argument("--k_mem", type=int, default=8)
    ap.add_argument("--turns", type=int, default=14)
    ap.add_argument("--keep_last", type=int, default=4)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dtype", type=str, default="bf16")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="logs/mem_encoder_v0.pt")
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm_dtype = pick_dtype(args.dtype)

    print(
        f"🚀 Training MemEncoder (digit-CE v2): model={args.model} device={device} "
        f"dtype={args.dtype} rep_layer={args.rep_layer} K={args.k_mem} "
        f"turns={args.turns} keep_last={args.keep_last}"
    )

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=lm_dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=lm_dtype)

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    d_model = model.config.hidden_size
    mem_enc = MemEncoder(d_model=d_model, K=args.k_mem, n_heads=8).to(device).float()
    opt = torch.optim.AdamW(mem_enc.parameters(), lr=args.lr, weight_decay=1e-4)

    embed = model.get_input_embeddings()
    ce = nn.CrossEntropyLoss()

    cand_ids_list, cand_last_ids, cand_lens = get_digit_candidates(tok)
    max_cand_len = max(cand_lens)

    # Precompute candidate embeddings & attention masks (NO grad)
    cand_emb = torch.zeros((10, max_cand_len, d_model), device=device, dtype=lm_dtype)
    cand_attn = torch.zeros((10, max_cand_len), device=device, dtype=torch.long)
    cand_pred_pos = torch.zeros((10,), device=device, dtype=torch.long)
    cand_pred_tok = torch.tensor(cand_last_ids, device=device, dtype=torch.long)

    with torch.no_grad():
        for d in range(10):
            ids = torch.tensor(cand_ids_list[d], device=device, dtype=torch.long).unsqueeze(0)  # (1,L)
            e = embed(ids).to(dtype=lm_dtype)  # (1,L,D)
            L = e.size(1)
            cand_emb[d, :L, :] = e.squeeze(0)
            cand_attn[d, :L] = 1
            # last token at index (prefix_len + L - 1) predicted by logits at (prefix_len + L - 2)
            # we’ll add prefix_len later, so store (L - 2) offset here
            cand_pred_pos[d] = (L - 2)

    running_correct = 0
    running_total = 0

    for step in range(1, args.steps + 1):
        removed_texts, prompts, gold_digits = [], [], []
        for _ in range(args.batch):
            rt, pr, gd = make_episode(rng, args.turns, args.keep_last)
            removed_texts.append(rt)
            prompts.append(pr)
            gold_digits.append(gd)

        batch = collate(tok, device, removed_texts, prompts, gold_digits, max_len=args.max_len)
        B = batch.prompt_ids.size(0)

        # 1) removed reps (no grad through frozen model)
        with torch.no_grad():
            out = model(
                input_ids=batch.rem_ids,
                attention_mask=batch.rem_attn,
                output_hidden_states=True,
                use_cache=False,
            )
            hs = out.hidden_states[args.rep_layer]  # (B,T,D)

        # 2) mem tokens (fp32 params)
        mem = mem_enc(hs, batch.rem_attn)  # (B,K,D) fp32
        if not torch.isfinite(mem).all():
            print("⚠️ mem has NaN/Inf — skipping step")
            opt.zero_grad(set_to_none=True)
            continue
        mem = mem.to(dtype=lm_dtype)

        # 3) prompt embeds (no grad needed)
        with torch.no_grad():
            prompt_emb_padded = embed(batch.prompt_ids).to(dtype=lm_dtype)  # (B,Tp,D)

        opt.zero_grad(set_to_none=True)

        step_loss_sum = 0.0
        step_correct = 0

        # KEY FIX: process ONE example at a time; backward immediately to avoid keeping graphs for B*10 forwards.
        for i in range(B):
            Lp = int(batch.prompt_lens[i].item())
            gold = batch.gold_digits[i:i+1]  # (1,)

            # Compute mem_i per-example so each backward has its own graph
            hs_i = hs[i:i+1, :, :]            # (1,T,D) no grad through frozen model
            attn_i = batch.rem_attn[i:i+1, :] # (1,T)

            mem_i = mem_enc(hs_i, attn_i)     # (1,K,D) fp32, graph is unique to this i
            if not torch.isfinite(mem_i).all():
                # skip this example (rare)
                continue
            mem_i = mem_i.to(dtype=lm_dtype)  # cast for LM forward

            prompt_i = prompt_emb_padded[i:i+1, :Lp, :]   # (1,Lp,D) no grad

            prefix = torch.cat([mem_i, prompt_i], dim=1)  # (1, K+Lp, D)
            prefix_len = prefix.size(1)

            # Expand prefix to 10 candidates
            prefix10 = prefix.expand(10, -1, -1)  # (10, prefix_len, D)

            # Full inputs_embeds for this example: [prefix] + [candidate]
            full_emb = torch.cat([prefix10, cand_emb], dim=1)  # (10, prefix_len+max_cand_len, D)

            # Attention mask: prefix all ones, candidate per-digit (handles variable cand length)
            attn_prefix = torch.ones((10, prefix_len), device=device, dtype=torch.long)
            full_attn = torch.cat([attn_prefix, cand_attn], dim=1)  # (10, T)

            out2 = model(
                inputs_embeds=full_emb,
                attention_mask=full_attn,
                use_cache=False,
            )
            logp = out2.logits.log_softmax(dim=-1)  # (10, T, V)

            # gather per-digit score = log p(last_token | prefix + candidate_prefix)
            # last token predicted by position (prefix_len + (L-2)) where L is cand length
            pos = (prefix_len + cand_pred_pos).to(device)  # (10,)
            row = torch.arange(10, device=device)

            scores_i = logp[row, pos, cand_pred_tok]  # (10,)
            loss_i = ce(scores_i.unsqueeze(0), gold)

            # backward immediately (accumulate grads across examples)
            loss_i.backward()

            step_loss_sum += float(loss_i.item())
            pred = int(scores_i.argmax().item())
            step_correct += int(pred == int(gold.item()))

            # free per-example refs ASAP
            del out2, logp, full_emb, full_attn, prefix10, prefix, scores_i, loss_i

        torch.nn.utils.clip_grad_norm_(mem_enc.parameters(), args.grad_clip)
        opt.step()

        running_correct += step_correct
        running_total += B

        if step % args.log_every == 0 or step == 1:
            acc = 100.0 * running_correct / max(1, running_total)
            avg_loss = step_loss_sum / max(1, B)
            print(f"[step {step}/{args.steps}] loss={avg_loss:.4f} digit-acc={acc:.2f}%")

    torch.save(mem_enc.state_dict(), args.save)
    print(f"✅ saved to {args.save}")


if __name__ == "__main__":
    main()
