import json, random
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from baselines.context_budget import Block, count_tokens, apply_discard, apply_extractive_kv_summary

MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_TOKENS = 160       # keep low so compression triggers quickly
TURNS = 14
SEED = 7

def final_answer(model, tok, blocks, question):
    prompt = (
        "Answer the user question using ONLY the provided context.\n"
        "If the answer is not present, reply with: NOT_FOUND\n\n"
        + "\n".join([f"[{b.kind}]\n{b.text}" for b in blocks])
        + f"\n\n[USER]\n{question}\n\n[ASSISTANT]\n"
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    in_len = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    gen = out[0][in_len:]
    text = tok.decode(gen, skip_special_tokens=True).strip()
    return text

def run_both():
    random.seed(SEED)

    qconf = BitsAndBytesConfig(load_in_4bit=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="auto", quantization_config=qconf, trust_remote_code=True
    )

    # Generate deterministic key-value facts
    # Values are "hard-to-guess" tokens; model must copy from context
    keys = []
    for i in range(1, TURNS + 1):
        val = f"{random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{random.randint(10,99)}-{random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{random.randint(10,99)}"
        keys.append((f"KEY{i:02d}", val))

    # Protected context (never discarded)
    base_blocks = [
        Block("KEEP", "RULE: Some blocks may be discarded or compressed if context exceeds the token budget."),
        Block("KEEP", "RULE: SCRATCH blocks contain key-value facts in the form KEYxx = VALUE."),
        Block("KEEP", "RULE: You must answer by copying the VALUE exactly (verbatim) from the context."),
    ]

    def simulate(method: str):
        blocks = list(base_blocks)
        log = {"method": method, "max_tokens": MAX_TOKENS, "turns": [], "target_key": "KEY01", "target_value": keys[0][1]}

        for t, (k, v) in enumerate(keys, start=1):
            # Add a new compressible SCRATCH fact each turn
            blocks.append(Block("SCRATCH", f"Turn {t} log: user message processed. Internal notes recorded. {k} = {v}"))

            before = count_tokens(tok, blocks)

            if method == "discard":
                blocks2 = apply_discard(tok, blocks, MAX_TOKENS)
            elif method == "summary":
                # summarize compressible history into one summary block
                blocks2 = apply_extractive_kv_summary(tok, blocks, MAX_TOKENS)
            else:
                raise ValueError("method must be discard or summary")

            after = count_tokens(tok, blocks2)
            blocks = blocks2

            log["turns"].append({
                "turn": t,
                "tokens_before": before,
                "tokens_after": after,
                "n_blocks": len(blocks),
                "k_added": k,
            })

        # Ask for the earliest key; discard should often have lost it
        ans = final_answer(model, tok, blocks, "What is the VALUE for KEY01? Reply with ONLY the VALUE. If missing, reply with ONLY NOT_FOUND.")
        ans0 = (ans or "").splitlines()[0].strip()

        log["final_answer_raw"] = ans
        log["final_answer"] = ans0
        log["correct"] = (ans0 == keys[0][1])
        return log

    out_discard = simulate("discard")
    out_summary = simulate("summary")

    print("discard correct:", out_discard["correct"], "answer:", out_discard["final_answer"])
    print("summary  correct:", out_summary["correct"], "answer:", out_summary["final_answer"])

    with open("logs/toy_v1_discard.json", "w", encoding="utf-8") as f:
        json.dump(out_discard, f, indent=2)
    with open("logs/toy_v1_summary.json", "w", encoding="utf-8") as f:
        json.dump(out_summary, f, indent=2)

if __name__ == "__main__":
    run_both()
