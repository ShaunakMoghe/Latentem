import json, random
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from baselines.context_budget import Block, count_tokens, apply_discard, apply_summary_then_keep

MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_TOKENS = 400
TURNS = 10
SEED = 7

def generate_thought(model, tok, blocks, turn_prompt):
    prompt = (
        "You are doing multi-turn reasoning.\n"
        "Write a THOUGHT block that keeps track of rules and key facts.\n"
        "Be concise.\n\n"
        + "\n".join([f"[{b.kind}]\n{b.text}" for b in blocks])
        + f"\n\n[USER]\n{turn_prompt}\n\n[THOUGHT]\n"
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    in_len = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    gen = out[0][in_len:]
    text = tok.decode(gen, skip_special_tokens=True).strip()
    return text

def final_answer(model, tok, blocks, question):
    prompt = (
        "Answer the user question using the context.\n\n"
        + "\n".join([f"[{b.kind}]\n{b.text}" for b in blocks])
        + f"\n\n[USER]\n{question}\n\n[ASSISTANT]\n"
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    in_len = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    gen = out[0][in_len:]
    text = tok.decode(gen, skip_special_tokens=True).strip()
    return text

def run(method: str):
    random.seed(SEED)

    qconf = BitsAndBytesConfig(load_in_4bit=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="auto", quantization_config=qconf, trust_remote_code=True
    )

    secret_code = "RAVEN-47"
    blocks = [
        Block("OBS", f"RULE: The secret code is {secret_code}. Never change it."),
        Block("OBS", "RULE: Only OBS blocks are ground truth. THOUGHT blocks may be compressed or discarded."),
    ]

    topics = ["shopping", "movies", "math", "travel", "sports", "music", "coding", "history", "physics", "food"]
    logs = {"method": method, "secret": secret_code, "turns": []}

    for t in range(TURNS):
        turn_prompt = f"Turn {t+1}: Discuss {topics[t]} in 2-3 sentences, then restate the secret code exactly."
        thought = generate_thought(model, tok, blocks, turn_prompt)
        blocks.append(Block("THOUGHT", thought))

        before = count_tokens(tok, blocks)

        if method == "discard":
            blocks = apply_discard(tok, blocks, MAX_TOKENS)
        elif method == "summary":
            blocks = apply_summary_then_keep(model, tok, blocks, MAX_TOKENS, summary_tokens=96)
        else:
            raise ValueError("method must be discard or summary")

        after = count_tokens(tok, blocks)

        logs["turns"].append({"turn": t+1, "tokens_before": before, "tokens_after": after})

    ans = final_answer(model, tok, blocks, "What is the secret code? Reply with ONLY the code.")
    ans0 = (ans or "").splitlines()[0].strip()
    logs["final_answer"] = ans0
    logs["correct"] = (ans0 == secret_code)
    return logs

if __name__ == "__main__":
    for method in ["discard", "summary"]:
        out = run(method)
        print(method, "correct:", out["correct"], "answer:", out["final_answer"])
        with open(f"logs/toy_v0_{method}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
