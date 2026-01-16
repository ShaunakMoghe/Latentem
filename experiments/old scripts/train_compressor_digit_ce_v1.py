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
        q_idx = rng.randint(0, removed_n - 1)
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

    # For scoring candidates, we will use candidates like " {d}" (no special tokens).
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


def get_digit_candidates(tok) -> Tuple[List[List[int]], List[int]]:
    """
    Returns:
      cand_ids_list[d] = token ids for " {d}" with add_special_tokens=False
      cand_last_ids[d] = last token id of that candidate
    """
    cand_ids_list = []
    cand_last_ids = []
    for d in range(10):
        ids = tok(" " + str(d), add_special_tokens=False).input_ids
        cand_ids_list.append(ids)
        cand_last_ids.append(ids[-1])
    return cand_ids_list, cand_last_ids


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
    ap.add_argument("--cand_microbatch", type=int, default=16)  # candidate forward microbatch (reduce if OOM)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm_dtype = pick_dtype(args.dtype)

    print(
        f"🚀 Training MemEncoder (digit-CE): model={args.model} device={device} "
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

    def rep_layer_index(hidden_states):
        L = len(hidden_states)
        return args.rep_layer if args.rep_layer >= 0 else (L + args.rep_layer)

    cand_ids_list, cand_last_ids = get_digit_candidates(tok)
    cand_lens = [len(x) for x in cand_ids_list]
    max_cand_len = max(cand_lens)

    # Precompute candidate embeddings in LM dtype (no grad needed)
    cand_embeds = []
    for ids in cand_ids_list:
        t = torch.tensor(ids, device=device, dtype=torch.long).unsqueeze(0)  # (1,L)
        cand_embeds.append(embed(t).to(dtype=lm_dtype))  # (1,L,D)

    ce = nn.CrossEntropyLoss()

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

        # 1) removed reps
        with torch.no_grad():
            out = model(
                input_ids=batch.rem_ids,
                attention_mask=batch.rem_attn,
                output_hidden_states=True,
                use_cache=False,
            )
            hs = out.hidden_states[rep_layer_index(out.hidden_states)]  # (B,T,D)

        # 2) mem tokens (fp32)
        mem = mem_enc(hs, batch.rem_attn)  # (B,K,D) fp32
        if not torch.isfinite(mem).all():
            print("⚠️ mem has NaN/Inf — skipping step")
            opt.zero_grad(set_to_none=True)
            continue
        mem = mem.to(dtype=lm_dtype)  # cast for LM

        # 3) build (B*10) sequences: [mem_i] + [prompt_i (trimmed)] + [cand_d]
        seq_emb_list = []
        seq_attn_list = []
        pred_pos_list = []
        pred_tok_list = []

        with torch.no_grad():
            prompt_emb_padded = embed(batch.prompt_ids).to(dtype=lm_dtype)  # (B,Tp,D)

        for i in range(B):
            Lp = int(batch.prompt_lens[i].item())
            prompt_i = prompt_emb_padded[i:i+1, :Lp, :]  # (1,Lp,D)
            mem_i = mem[i:i+1, :, :]  # (1,K,D)

            for d in range(10):
                cand_e = cand_embeds[d]  # (1,Lc,D)
                Lc = cand_e.size(1)

                seq = torch.cat([mem_i, prompt_i, cand_e], dim=1)  # (1, K+Lp+Lc, D)
                seq_emb_list.append(seq.squeeze(0))

                attn = torch.ones(seq.size(1), device=device, dtype=torch.long)
                seq_attn_list.append(attn)

                # We score ONLY the last token of the candidate (digit token).
                # token at index (K+Lp+Lc-1) is predicted by logits at (K+Lp+Lc-2).
                pred_pos_list.append((args.k_mem + Lp + Lc - 2))
                pred_tok_list.append(cand_last_ids[d])

        # Microbatch forward to avoid OOM: evaluate candidate sequences in chunks
        pred_pos_all = torch.tensor(pred_pos_list, device=device, dtype=torch.long)  # (N,)
        pred_tok_all = torch.tensor(pred_tok_list, device=device, dtype=torch.long)  # (N,)
        N = len(seq_emb_list)
        mb = int(args.cand_microbatch)

        scores_chunks = []
        for start in range(0, N, mb):
            end = min(N, start + mb)

            # pad only this chunk
            chunk_emb_list = seq_emb_list[start:end]
            chunk_attn_list = seq_attn_list[start:end]
            maxT = max(x.size(0) for x in chunk_emb_list)
            D = chunk_emb_list[0].size(1)

            chunk_emb = torch.zeros((end - start, maxT, D), device=device, dtype=lm_dtype)
            chunk_attn = torch.zeros((end - start, maxT), device=device, dtype=torch.long)

            for r, (e, a) in enumerate(zip(chunk_emb_list, chunk_attn_list)):
                T = e.size(0)
                chunk_emb[r, :T, :] = e
                chunk_attn[r, :T] = a

            out2 = model(
                inputs_embeds=chunk_emb,
                attention_mask=chunk_attn,
                use_cache=False,
            )
            logits = out2.logits  # (mb, maxT, V)

            # gather only the needed log-prob for each row
            logp = logits.log_softmax(dim=-1)
            pos = pred_pos_all[start:end]   # (mb,)
            tok = pred_tok_all[start:end]   # (mb,)
            row = torch.arange(end - start, device=device)

            scores_chunk = logp[row, pos, tok]  # (mb,)
            scores_chunks.append(scores_chunk)

            # free references ASAP
            del chunk_emb, chunk_attn, out2, logits, logp, scores_chunk

        scores_flat = torch.cat(scores_chunks, dim=0)  # (N,)
        scores = scores_flat.view(B, 10)  # (B,10)

        loss = ce(scores, batch.gold_digits)

        if not torch.isfinite(loss):
            print(f"⚠️ loss is NaN/Inf at step {step} — skipping update")
            opt.zero_grad(set_to_none=True)
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mem_enc.parameters(), args.grad_clip)
        opt.step()

        # accuracy
        with torch.no_grad():
            pred = scores.argmax(dim=1)
            running_correct += int((pred == batch.gold_digits).sum().item())
            running_total += int(B)

        if step % args.log_every == 0 or step == 1:
            acc = 100.0 * running_correct / max(1, running_total)
            print(f"[step {step}/{args.steps}] loss={loss.item():.4f} digit-acc={acc:.2f}%")

    torch.save(mem_enc.state_dict(), args.save)
    print(f"✅ saved to {args.save}")


if __name__ == "__main__":
    main()
