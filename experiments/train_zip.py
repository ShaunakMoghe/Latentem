"""
Training Script for Latent Zip Architecture (Soft Prompt Version).
Task: Compress 'Context' -> Predict 'Target'.
"""
import argparse
import random
import torch
import sys
import os
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from src.latent_zip import LatentZipSoftPrompt

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- DATA (Reused from train_curriculum.py) ---
KEYS = [f"KEY{i:02d}" for i in range(100)]
DIGITS = list(range(10))

class Episode:
    def __init__(self, context_text, target_text):
        self.context_text = context_text
        self.target_text = target_text

def make_episode(rng, turns, keep_last):
    # Create a KV sequence
    kv = [(rng.choice(KEYS), rng.choice(DIGITS)) for _ in range(turns)]
    
    # QA Task: Context -> Zip. Target -> Question + Answer.
    context_kvs = kv
    context_lines = [f"{k} = {v}" for k, v in context_kvs]
    context_text = "\n".join(context_lines)
    
    # Pick a random query from context
    q_key, q_val = rng.choice(context_kvs)
    
    # Target: "Question: what is value of KEYxx?\nAnswer: Y"
    # We want the model to see [SoftPrompts] + [Question] -> generate [Answer]
    target_text = f"Question: what is the value of {q_key}?\nAnswer: {q_val}"
    
    return Episode(context_text, target_text)

def build_batch(tok, episodes, device):
    # 1. Encode Context (For Encoder)
    ctx_texts = [e.context_text for e in episodes]
    ctx_enc = tok(ctx_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    
    # 2. Encode Target (For Decoder Inputs/Labels)
    tgt_texts = [e.target_text for e in episodes]
    tgt_enc = tok(tgt_texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    
    input_ids = tgt_enc.input_ids
    labels = input_ids.clone()
    labels[input_ids == tok.pad_token_id] = -100 # Mask padding
    
    # Mask Question part
    # We find the index of "Answer:" and mask everything before it.
    # Tokenizer might split "Answer:" differently depending on context.
    # Safe way: Encode "Question... Answer:" part separately?
    for i, text in enumerate(tgt_texts):
        # Split at "Answer:"
        # Note: "Answer:" might not be a single token.
        # Find string index
        answer_idx = text.find("Answer:")
        if answer_idx != -1:
            # Add len("Answer:") to start predicting AFTER the colon?
            # Or predict "Answer" and ":"?
            # Let's predict the VALUE.
            value_start_idx = answer_idx + len("Answer:")
            prefix_text = text[:value_start_idx]
            
            # Encode prefix
            prefix_ids = tok(prefix_text, add_special_tokens=False).input_ids
            # Len of prefix in tokens
            # We need to account for BOS if present?
            # Qwen tokenizer usually handles this.
            # Let's just find the approximate length.
            mask_len = len(prefix_ids)
            # Mask [0 : mask_len]
            # Be careful with BOS.
            if mask_len < labels.shape[1]:
                labels[i, :mask_len] = -100
                
    return ctx_enc.input_ids, input_ids, labels

# --- MAIN ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--lr", type=float, default=2e-4) # Slightly higher LR for Soft Prompts
    args = ap.parse_args()
    
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Model
    print(f"Loading LatentZipSoftPrompt: {args.model}...")
    model = LatentZipSoftPrompt(args.model, num_zip_tokens=32, freeze_decoder=False).to(device).to(dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    
    print("🚀 Starting Training (Soft Prompt)...")
    rng = random.Random(42)
    
    import time
    start_time = time.time()
    
    for step in range(1, args.steps + 1):
        # 1. Data
        # Capacity Test: 2 Turns Fixed.
        episodes = [make_episode(rng, 2, 5) for _ in range(8)] 
        input_ids_enc, input_ids_dec, labels = build_batch(tok, episodes, device)
        
        # 2. Forward
        # Output Logits shape: (B, K + S, V)
        outputs = model(input_ids_enc, input_ids_dec)
        logits = outputs.logits
        
        # 3. Loss
        # We need to shift labels.
        # The logits correspond to [SoftPrompts, TargetTokens].
        # We only want to predict TargetTokens.
        # The first K logits predict the first Target token?
        # Standard: Logits[i] predicts Token[i+1].
        # Input: [SP_1 ... SP_K, T_1, T_2 ... T_N]
        # Logits: [P(SP_2), .. P(T_1), P(T_2) ... P(T_N+1)]
        # We want to match P(T_1) with T_1? No. P(T_1) comes from SP_K.
        # So we align Logits[K-1 : -1] with Labels[0 : N]
        
        num_zip = model.encoder.num_zip_tokens
        
        # Logits responsible for predicting target:
        # The logit at index (K-1) predicts the first token of Target.
        # The logit at index (K+S-2) predicts the last token of Target.
        # Let's verify shift.
        # Causal LM: input[i] -> predict[i+1].
        # Inputs: [SP, T]
        # We want to predict [T].
        # So we want logits from [SP[-1], T[:-1]] to predict T.
        # SP[-1] is at index K-1.
        # T[:-1] ends at end-2.
        
        # So shift_logits = logits[:, num_zip-1 : -1, :]
        # shift_labels = labels
        
        shift_logits = logits[:, num_zip-1 : -1, :].contiguous()
        shift_labels = labels.contiguous()
        
        # Ensure shapes match (might differ by 1 if not careful)
        # logits length = K + S
        # labels length = S
        # shift_logits length = (K+S) - (num_zip-1) - 1 = S. Correct.
        
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        
        # Accuracy
        with torch.no_grad():
            preds = torch.argmax(shift_logits, dim=-1)
            mask = shift_labels != -100
            correct = (preds == shift_labels) & mask
            accuracy = correct.sum().float() / mask.sum().float()
        
        # 4. Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        
        if step % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}/{args.steps} | Loss: {loss.item():.4f} | Acc: {accuracy.item():.4f} | Time: {elapsed:.1f}s")
            
        # Generation Check (Every 20 steps)
        if step % 20 == 0:
             model.eval()
             with torch.no_grad():
                 # Prompt: Question without Answer
                 target_text = episodes[0].target_text
                 # Split at "Answer:"
                 prompt_text = target_text.split("Answer:")[0] + "Answer:"
                 
                 prompt_ids = tok(prompt_text, return_tensors="pt").input_ids.to(device)
                 
                 try:
                     gen_ids = model.generate(
                         input_ids_encode=input_ids_enc[0:1],
                         prompt_ids=prompt_ids,
                         max_new_tokens=5
                     )
                     gen_text = tok.decode(gen_ids[0], skip_special_tokens=True)
                     print(f"\n--- Gen Step {step} ---")
                     print(f"Target: {target_text.strip()}")
                     print(f"Pred:   {gen_text.strip()}")
                     print("-----------------------")
                 except Exception as e:
                     print(f"GEN FAILED: {e}")
             model.train()

    print("Training Complete.")

if __name__ == "__main__":
    main()
