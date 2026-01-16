# experiments/eval_budget_sweep_v0.py
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.latent_mem import MemEncoder as TrainMemEncoder
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict # <--- NEW

# -------------------------
#  Utilities
# -------------------------

def parse_dtype(s: str):
    s = s.lower().strip()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {s}")


def truncate_left(ids_1d: torch.Tensor, budget: int) -> torch.Tensor:
    """Keep last `budget` tokens."""
    if ids_1d.numel() <= budget:
        return ids_1d
    return ids_1d[-budget:]


def _topk_digits(digits: List[int], scores: List[float], k: int = 5) -> List[Tuple[int, float]]:
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(digits[i], float(scores[i])) for i in idx]


def get_digit_candidates(tok, mode: str) -> Tuple[List[int], List[torch.Tensor]]:
    """
    mode:
      - space_digit: candidates are " {d}"
      - digit_only : candidates are "{d}"
    """
    digits: List[int] = []
    seqs: List[torch.Tensor] = []
    for d in range(10):
        s = (" " + str(d)) if mode == "space_digit" else str(d)
        ids = tok(s, add_special_tokens=False).input_ids
        digits.append(d)
        seqs.append(torch.tensor(ids, dtype=torch.long))
    return digits, seqs


@torch.inference_mode()
def reps_from_text(model, tok, text: str, rep_layer: int, device: str, max_len: int):
    enc = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        add_special_tokens=True,
    )
    ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[rep_layer]  # (1,T,D)
    return hs, attn


@torch.inference_mode()
def score_candidates_from_prompt_embeds(
    model,
    prompt_embeds: torch.Tensor,           # (1,T,D)
    cand_token_seqs: List[torch.Tensor],   # list of 1D token id tensors (variable length)
) -> Tuple[List[float], float]: # Return logits list AND NLL of best token
    """
    Compute log p(candidate | prompt).
    """
    if prompt_embeds.dim() != 3 or prompt_embeds.size(0) != 1:
        raise ValueError(f"prompt_embeds must be (1,T,D). Got {tuple(prompt_embeds.shape)}")

    device = prompt_embeds.device
    emb_layer = model.get_input_embeddings()

    lens = torch.tensor([int(x.numel()) for x in cand_token_seqs], device=device, dtype=torch.long)
    if lens.numel() == 0:
        return [], 0.0

    maxL = int(lens.max().item())
    pad_id = int(getattr(model.config, "eos_token_id", 0) or 0)

    cand_ids = torch.full((len(cand_token_seqs), maxL), pad_id, device=device, dtype=torch.long)
    for i, seq in enumerate(cand_token_seqs):
        li = int(seq.numel())
        if li > 0:
            cand_ids[i, :li] = seq.to(device)

    cand_emb = emb_layer(cand_ids).to(dtype=prompt_embeds.dtype)  # (N,maxL,D)

    N = cand_ids.size(0)
    T = prompt_embeds.size(1)

    prefix = prompt_embeds.expand(N, -1, -1)           # (N,T,D)
    full_emb = torch.cat([prefix, cand_emb], dim=1)    # (N,T+maxL,D)

    attn = torch.ones(full_emb.shape[:2], device=device, dtype=torch.long)
    pos_ids = torch.arange(full_emb.size(1), device=device, dtype=torch.long).unsqueeze(0).expand(N, -1)

    logits = model(
        inputs_embeds=full_emb,
        attention_mask=attn,
        position_ids=pos_ids,
        use_cache=False,
    ).logits  # (N,T+maxL,V)

    logp = logits.log_softmax(dim=-1)

    pos = (T - 1) + torch.arange(maxL, device=device, dtype=torch.long)   # (maxL,)
    pos = pos.unsqueeze(0).expand(N, -1)                                  # (N,maxL)

    token_logp = logp[torch.arange(N, device=device).unsqueeze(1), pos, cand_ids]  # (N,maxL)
    mask = (torch.arange(maxL, device=device).unsqueeze(0) < lens.unsqueeze(1))   # (N,maxL)
    scores = (token_logp * mask).sum(dim=1)  # (N,)
    
    # METRIC 1: NLL (Negative Log Likelihood) of the candidates
    # We return the scores (log probs). 
    # The 'best match' NLL is -1 * max(scores)
    
    return [float(s.item()) for s in scores]


# -------------------------
#  MemEncoder loading
# -------------------------

def load_mem_encoder_from_ckpt(path: str, d_model: int, device: str) -> Tuple[nn.Module, Dict[str, Any], Optional[Dict]]:
    raw = torch.load(path, map_location="cpu")
    
    # Handle wrapped checkpoint
    if isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
        meta = raw.get("meta", {})
        lora_sd = raw.get("lora_state_dict", None) # <--- NEW
    else:
        sd = raw
        meta = {}
        lora_sd = None

    if "comp.mem_queries" not in sd:
        raise KeyError("Checkpoint missing 'comp.mem_queries' (unexpected state_dict format).")

    K = int(sd["comp.mem_queries"].shape[0])
    enc = TrainMemEncoder(d_model=d_model, K=K, n_heads=8)
    ik = enc.load_state_dict(sd, strict=False)
    
    enc = enc.to(device=device).float().eval()
    meta.update({"K": K, "d_model": d_model})
    return enc, meta, lora_sd


# -------------------------
#  Episode / baselines
# -------------------------

@dataclass
class Episode:
    removed_text: str
    kept_text: str
    query_text: str
    target_key: str
    target_val: str
    removed_pairs: List[Tuple[str, str]]


def make_episode(
    turns: int,
    keep_last: int,
    force_mem_target: bool,
    rng: random.Random,
    digit_first: bool,
) -> Episode:
    keys = [f"KEY{i:02d}" for i in range(1, turns + 1)]
    vals = [str(rng.randint(0, 9)) for _ in range(turns)]
    lines = [f"Turn {i}: {k} = {v}." for i, (k, v) in enumerate(zip(keys, vals), start=1)]

    cut = turns - keep_last
    removed_pairs = list(zip(keys[:cut], vals[:cut]))

    if force_mem_target and removed_pairs:
        target_key, target_val = rng.choice(removed_pairs)
    else:
        target_key, target_val = rng.choice(list(zip(keys, vals)))

    removed_text = "\n".join(lines[:cut]) if cut > 0 else "(nothing removed)"
    kept_text = "\n".join(lines[cut:])

    if digit_first:
        query_text = f"\nQuestion: What is {target_key}?\nAnswer: "
    else:
        query_text = f"\nQuestion: What is {target_key}?\nAnswer:"

    return Episode(
        removed_text=removed_text,
        kept_text=kept_text,
        query_text=query_text,
        target_key=target_key,
        target_val=target_val,
        removed_pairs=removed_pairs,
    )


def build_text_summary_to_fit(tok, removed_pairs: List[Tuple[str, str]], kept_text: str, query_text: str, budget: int) -> str:
    base = kept_text + query_text
    base_len = len(tok(base, add_special_tokens=False).input_ids)
    if base_len >= budget:
        return ""  # no room

    summary_pairs = []
    for (k, v) in removed_pairs:
        summary_pairs.append((k, v))
        summ = "Summary: " + " ".join([f"{kk}={vv};" for (kk, vv) in summary_pairs]) + "\n"
        total = summ + base
        if len(tok(total, add_special_tokens=False).input_ids) > budget:
            summary_pairs.pop()
            break

    if not summary_pairs:
        return ""
    return "Summary: " + " ".join([f"{kk}={vv};" for (kk, vv) in summary_pairs]) + "\n"


# -------------------------
#  Evaluation
# -------------------------

# METRIC 2: Linear Probe Class
class Probe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.linear(x)

@torch.inference_mode()
def eval_once(
    model,
    tok,
    digits: List[int],
    cand_token_seqs: List[torch.Tensor],
    enc: nn.Module,
    rep_layer: int,
    K: int,
    ep: Episode,
    budgets: List[int],
    device: str,
    dtype: torch.dtype,
    max_len: int,
    deep_debug: bool,
    debug_budget: int,
    mem_norm: bool,
    probe: Optional[Probe] = None, # Hook for probe
):
    kept_query_text = ep.kept_text + ep.query_text
    results = {} 

    emb_layer = model.get_input_embeddings()

    # Compute removed reps once
    reps, rem_attn = reps_from_text(model, tok, ep.removed_text, rep_layer, device, max_len)
    mem_learned = enc(reps, rem_attn).to(dtype=dtype)  # (1,K,D)
    
    # METRIC 2: Probe check (Does the memory contain the digit?)
    probe_pred = -1
    if probe is not None:
        # Simple pooling or slot-selection for probing. Here we try mean-pool for simplicity
        with torch.no_grad():
            probe_logits = probe(mem_learned.float().mean(dim=1)) # (1, 10)
            probe_pred = probe_logits.argmax(dim=-1).item()

    for b in budgets:
        do_print = deep_debug and (debug_budget < 0 or b == debug_budget)
        gold_d = int(ep.target_val)

        # -----------------
        # Discard baseline
        # -----------------
        ids = tok(kept_query_text, add_special_tokens=False).input_ids
        ids = truncate_left(torch.tensor(ids, device=device), b)
        discard_emb = emb_layer(ids.unsqueeze(0)).to(dtype=dtype)
        scores_discard = score_candidates_from_prompt_embeds(model, discard_emb, cand_token_seqs)
        pred_discard = digits[max(range(len(scores_discard)), key=lambda i: scores_discard[i])]
        # METRIC 1: NLL 
        nll_discard = -1.0 * scores_discard[gold_d]

        # -----------------
        # Summary baseline
        # -----------------
        summ = build_text_summary_to_fit(tok, ep.removed_pairs, ep.kept_text, ep.query_text, b)
        summ_text = summ + kept_query_text
        ids_s = tok(summ_text, add_special_tokens=False).input_ids
        ids_s = truncate_left(torch.tensor(ids_s, device=device), b)
        summ_emb = emb_layer(ids_s.unsqueeze(0)).to(dtype=dtype)
        scores_summ = score_candidates_from_prompt_embeds(model, summ_emb, cand_token_seqs)
        pred_summ = digits[max(range(len(scores_summ)), key=lambda i: scores_summ[i])]
        nll_summ = -1.0 * scores_summ[gold_d]

        # -----------------
        # Latent baseline
        # -----------------
        keep_budget = max(1, b - K)
        ids_k = tok(kept_query_text, add_special_tokens=False).input_ids
        ids_k = truncate_left(torch.tensor(ids_k, device=device), keep_budget)
        kept_emb = emb_layer(ids_k.unsqueeze(0)).to(dtype=dtype)

        mem = mem_learned
        if mem_norm:
            ref_norm = kept_emb.norm(dim=-1).mean()
            mem_n = mem.norm(dim=-1).mean() + 1e-9
            scale = (ref_norm / mem_n).clamp(0.01, 100.0)
            mem = mem * scale

        latent_emb = torch.cat([mem, kept_emb], dim=1)
        scores_latent = score_candidates_from_prompt_embeds(model, latent_emb, cand_token_seqs)
        pred_latent = digits[max(range(len(scores_latent)), key=lambda i: scores_latent[i])]
        nll_latent = -1.0 * scores_latent[gold_d]

        results[b] = {
            "discard_acc": 1 if pred_discard == gold_d else 0,
            "summ_acc": 1 if pred_summ == gold_d else 0,
            "latent_acc": 1 if pred_latent == gold_d else 0,
            "discard_nll": nll_discard,
            "summ_nll": nll_summ,
            "latent_nll": nll_latent,
            "probe_acc": 1 if probe_pred == gold_d else 0
        }

        if do_print:
            print(f"    [dbg][B{b}] NLL: Disc={nll_discard:.2f} Summ={nll_summ:.2f} Latent={nll_latent:.2f}")
            if probe is not None:
                print(f"    [dbg][B{b}] Probe Pred: {probe_pred} (Correct: {probe_pred==gold_d})")

    gold_digit = int(ep.target_val)
    return gold_digit, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--rep_layer", type=int, default=12)
    ap.add_argument("--k_mem", type=int, default=8)
    ap.add_argument("--turns", type=int, default=14)
    ap.add_argument("--keep_last", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--budgets", type=str, default="64,96,128")
    ap.add_argument("--dtype", type=str, default="bf16")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_len", type=int, default=256)

    ap.add_argument("--force_mem_target", action="store_true")
    ap.add_argument("--debug_first", type=int, default=0)
    ap.add_argument("--digit_first", action="store_true")
    ap.add_argument("--cand_mode", type=str, default="space_digit", choices=["space_digit", "digit_only"])
    ap.add_argument("--deep_debug", action="store_true")
    ap.add_argument("--debug_budget", type=int, default=-1)
    ap.add_argument("--no_mem_norm", action="store_true")
    ap.add_argument("--train_probe", action="store_true", help="Train a linear probe on the fly to check info content.")

    args = ap.parse_args()

    device = args.device
    dtype = parse_dtype(args.dtype)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    budgets = sorted(budgets)

    print(
        f"📏 Budget sweep: model={args.model} rep_layer={args.rep_layer} K={args.k_mem} budgets={budgets} "
        f"episodes={args.episodes} dtype={args.dtype} device={device}"
    )

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype=dtype, device_map=None)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=dtype, device_map=None)
    
    # 2. LOAD LORA IF PRESENT
    enc, enc_meta, lora_sd = load_mem_encoder_from_ckpt(args.ckpt, d_model=model.config.hidden_size, device=device)
    if lora_sd is not None:
        print("🚀 Detected LoRA weights! Applying adapters...")
        peft_config = LoraConfig(
            r=64, lora_alpha=128, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        set_peft_model_state_dict(model, lora_sd)
        print("✅ LoRA adapters loaded.")
    else:
        print("⚠️ No LoRA weights found in checkpoint. Using frozen base model.")
    
    model = model.to(device).eval()

    digits, cand_token_seqs = get_digit_candidates(tok, args.cand_mode)
    K = enc_meta["K"]

    # Optional: Train Probe on the fly
    probe = None
    if args.train_probe:
        print("🧪 Training diagnostic linear probe on 100 episodes...")
        probe = Probe(model.config.hidden_size, 10).to(device).float()
        p_opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        for _ in range(100):
            ep = make_episode(args.turns, args.keep_last, True, rng, args.digit_first)
            reps, rem_attn = reps_from_text(model, tok, ep.removed_text, args.rep_layer, device, args.max_len)
            with torch.no_grad():
                mem = enc(reps, rem_attn)
            logits = probe(mem.float().mean(dim=1))
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([int(ep.target_val)], device=device))
            p_opt.zero_grad()
            loss.backward()
            p_opt.step()
        print("✅ Probe trained.")

    metrics = {b: {"acc_d": 0, "acc_s": 0, "acc_l": 0, "nll_d": 0.0, "nll_s": 0.0, "nll_l": 0.0, "probe": 0} for b in budgets}

    for ep_i in range(args.episodes):
        ep = make_episode(args.turns, args.keep_last, args.force_mem_target, rng, digit_first=args.digit_first)
        deep = (args.deep_debug and args.debug_first and ep_i < args.debug_first)

        gold, res = eval_once(
            model=model, tok=tok, digits=digits, cand_token_seqs=cand_token_seqs,
            enc=enc, rep_layer=args.rep_layer, K=K, ep=ep, budgets=budgets,
            device=device, dtype=dtype, max_len=args.max_len,
            deep_debug=deep, debug_budget=args.debug_budget,
            mem_norm=(not args.no_mem_norm), probe=probe
        )

        for b in budgets:
            metrics[b]["acc_d"] += res[b]["discard_acc"]
            metrics[b]["acc_s"] += res[b]["summ_acc"]
            metrics[b]["acc_l"] += res[b]["latent_acc"]
            metrics[b]["nll_d"] += res[b]["discard_nll"]
            metrics[b]["nll_s"] += res[b]["summ_nll"]
            metrics[b]["nll_l"] += res[b]["latent_nll"]
            metrics[b]["probe"] += res[b]["probe_acc"]

    print("\n=== Results ===")
    for b in budgets:
        N = args.episodes
        print(f"Budget {b}:")
        print(f"  Acc: Discard={metrics[b]['acc_d']/N:.1%} Summary={metrics[b]['acc_s']/N:.1%} Latent={metrics[b]['acc_l']/N:.1%}")
        print(f"  NLL: Discard={metrics[b]['nll_d']/N:.3f} Summary={metrics[b]['nll_s']/N:.3f} Latent={metrics[b]['nll_l']/N:.3f}")
        if args.train_probe:
            print(f"  Probe Acc: {metrics[b]['probe']/N:.1%}")

if __name__ == "__main__":
    main()