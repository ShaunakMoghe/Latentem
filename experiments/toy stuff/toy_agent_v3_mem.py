import argparse
import random
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.latent_mem import LatentCompressor, LearnedProjection, attn_mask_to_key_padding
from src.prompt_with_mem import prepend_memory_to_input_ids


def build_kv_convo(turns: int, rng: random.Random) -> Tuple[str, str, str]:
    keys = [f"KEY{i:02d}" for i in range(1, turns + 1)]
    vals = [str(rng.randrange(0, 10)) for _ in range(turns)]

    # target in early part
    target_idx = rng.randrange(0, max(1, turns // 3))
    target_key = keys[target_idx]
    target_val = vals[target_idx]

    lines = [f"Turn {i}: {k} is {v}." for i, (k, v) in enumerate(zip(keys, vals), start=1)]
    history = "\n".join(lines)

    question = f"\nQuestion: What is {target_key}?\nAnswer:"
    gold = f" {target_val}"
    return history, question, gold


def token_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False).input_ids)


def truncate_history_to_budget(tok, history: str, question: str, budget: int) -> Tuple[str, str]:
    lines = history.splitlines()
    kept = []
    for line in reversed(lines):
        trial = "\n".join(reversed(kept + [line]))
        if token_len(tok, trial + question) <= budget:
            kept.append(line)
        else:
            break
    kept_history = "\n".join(reversed(kept))
    removed_history = "\n".join(lines[: len(lines) - len(kept)])
    return removed_history, kept_history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ckpt", type=str, default="logs/mem_encoder_v0.pt")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device)
    model.eval()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    rep_layer = ckpt["rep_layer"]
    D = ckpt["d_model"]
    K = ckpt["k_mem"]

    proj = LearnedProjection(D, use_layernorm=True, init_identity=False).to(args.device)
    comp = LatentCompressor(D, K=K, n_heads=8, dropout=0.0).to(args.device)
    proj.load_state_dict(ckpt["proj"])
    comp.load_state_dict(ckpt["compressor"])
    proj.eval()
    comp.eval()

    ok_discard = 0
    ok_mem = 0

    for ep in range(1, args.episodes + 1):
        history, question, gold = build_kv_convo(args.turns, rng)
        removed, kept = truncate_history_to_budget(tok, history, question, args.budget)

        # discard
        prompt_discard = kept + question
        ids = tok(prompt_discard, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
        attn = torch.ones_like(ids)
        out = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=4, do_sample=False, use_cache=True)
        text_discard = tok.decode(out[0], skip_special_tokens=True)
        ok_discard += int(gold.strip() in text_discard.split("Answer:")[-1])

        # latent memory
        rem_ids = tok(removed, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
        rem_attn = torch.ones_like(rem_ids)

        with torch.no_grad():
            out_rem = model(input_ids=rem_ids, attention_mask=rem_attn, output_hidden_states=True, use_cache=False)
            hs = out_rem.hidden_states[rep_layer]

        mem = comp(proj(hs), seq_mask=attn_mask_to_key_padding(rem_attn))

        kept_ids = tok(kept + question, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
        inputs_embeds, _ = prepend_memory_to_input_ids(model, tok, kept_ids, mem)
        attn2 = torch.ones((1, inputs_embeds.size(1)), dtype=torch.long, device=args.device)

        out2 = model.generate(inputs_embeds=inputs_embeds, attention_mask=attn2, max_new_tokens=4, do_sample=False, use_cache=True)
        text_mem = tok.decode(out2[0], skip_special_tokens=True)
        ok_mem += int(gold.strip() in text_mem.split("Answer:")[-1])

        if ep % 10 == 0:
            print(f"[{ep}/{args.episodes}] discard={ok_discard/ep*100:.1f}% mem={ok_mem/ep*100:.1f}%")

    print("\n===== RESULTS =====")
    print(f"discard acc: {ok_discard/args.episodes*100:.2f}%")
    print(f"latent-mem acc: {ok_mem/args.episodes*100:.2f}%")


if __name__ == "__main__":
    main()
