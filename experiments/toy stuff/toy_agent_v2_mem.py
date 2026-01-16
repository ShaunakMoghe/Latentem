import json, random, re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from baselines.context_budget import Block, count_tokens
from src.latent_mem import LatentCompressor
from src.prompt_with_mem import build_inputs_with_mem

MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_TOKENS = 160
TURNS = 14
SEED = 7
K_MEM = 8

def parse_value(val: str):
    m = re.match(r"^([A-Z])(\d{2})\-([A-Z])(\d{2})$", val.strip())
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3), int(m.group(4))

def final_answer(model, tok, blocks, question, mem=None):
    prompt = (
        "Answer using ONLY the provided context.\n"
        "Reply with ONLY the final integer, no words.\n\n"
        + "\n".join([f"[{b.kind}]\n{b.text}" for b in blocks])
        + f"\n\n[USER]\n{question}\n\n[ASSISTANT]\n"
    )

    gen_inputs = build_inputs_with_mem(model, tok, prompt, mem)
    in_len = gen_inputs["input_ids"].shape[1]

    out = model.generate(
        **gen_inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )

    gen_ids = out[0][in_len:] if out.shape[1] > in_len else out[0]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return text

def run():
    random.seed(SEED)

    qconf = BitsAndBytesConfig(load_in_4bit=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="auto", quantization_config=qconf, trust_remote_code=True
    )
    model.eval()

    d_model = model.config.hidden_size
    compressor = LatentCompressor(d_model=d_model, K=K_MEM).to(model.device)

    # load trained compressor if available
    try:
        ckpt = torch.load("logs/mem_v0.pt", map_location=model.device)
        compressor.load_state_dict(ckpt["compressor"])
        compressor.eval()
        print("loaded compressor from logs/mem_v0.pt")
    except Exception as e:
        print("warning: could not load logs/mem_v0.pt, using random compressor:", e)

    keys = []
    for i in range(1, TURNS + 1):
        val = f"{random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{random.randint(10,99)}-{random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{random.randint(10,99)}"
        keys.append((f"KEY{i:02d}", val))

    base_blocks = [
        Block("KEEP", "RULE: Some blocks may be discarded or compressed if context exceeds the token budget."),
        Block("KEEP", "RULE: SCRATCH blocks contain KEYxx = VALUE facts."),
    ]

    mem = None
    blocks = list(base_blocks)

    for t, (k, v) in enumerate(keys, start=1):
        blocks.append(Block("SCRATCH", f"Turn {t} log: processed message. {k} = {v}"))

        if count_tokens(tok, blocks) > MAX_TOKENS:
            compress_blocks = [b for b in blocks if b.kind in ("SCRATCH", "THOUGHT", "SUMMARY")]
            compress_text = "\n".join([f"[{b.kind}]\n{b.text}" for b in compress_blocks])
            inputs = tok(compress_text, return_tensors="pt").to(model.device)

            # Use hidden states (better than raw embeddings)
            with torch.no_grad():
                backbone = getattr(model, "model", None)
                out = backbone(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], return_dict=True)
                h = out.last_hidden_state.float()

            with torch.no_grad():
                mem = compressor(h, seq_mask=inputs["attention_mask"].bool())

            blocks = [b for b in blocks if b.kind == "KEEP"]
            blocks.append(Block("SUMMARY", f"Latent memory updated at turn {t}."))

    v1 = parse_value(keys[0][1])
    v2 = parse_value(keys[1][1])
    target = str(v1[1] + v2[1]) if (v1 and v2) else ""

    ans_raw = final_answer(model, tok, blocks, "Compute: (KEY01 first number) + (KEY02 first number). Reply with only the integer.", mem=mem)
    m = re.search(r"-?\d+", ans_raw)
    ans = m.group(0) if m else ""

    out = {"target": target, "answer": ans, "answer_raw": ans_raw, "correct": (ans == target)}
    print("mem correct:", out["correct"], "answer:", ans_raw, "target:", target)

    with open("logs/toy_v2_mem.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    run()
