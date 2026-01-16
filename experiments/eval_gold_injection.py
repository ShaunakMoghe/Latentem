# experiments/eval_gold_injection.py
import argparse
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ckpt", required=True, help="Path to LoRA checkpoint")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # 1. Load Model + LoRA
    print(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device)
    
    # Load your trained LoRA adapter
    # We need to extract just the adapter config/weights from your checkpoint dictionary
    print(f"Loading LoRA from {args.ckpt}...")
    ckpt_data = torch.load(args.ckpt, map_location=args.device)
    
    # PEFT loading hack: save temp adapter to load it cleanly
    if "lora_state_dict" in ckpt_data:
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        model = get_peft_model(base, config)
        set_peft_model_state_dict(model, ckpt_data["lora_state_dict"])
        print("✅ LoRA weights loaded.")
    else:
        print("❌ No LoRA weights found! Testing frozen model.")
        model = base

    model.eval()
    embed = model.get_input_embeddings()

    # 2. Run Test
    correct_digit = 0
    correct_kv = 0
    total = 0
    
    print("\n--- Running Golden Injection Test ---")
    print("We will feed the model PERFECT embeddings and see if it can answer.")

    for i in range(50): # 50 examples
        # Create a sample episode
        target_val = random.randint(0, 9)
        target_key = f"KEY{random.randint(1, 10):02d}"
        
        # Prompt (No context, just the question)
        prompt = f"\nQuestion: What is {target_key}?\nAnswer:"
        prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(args.device)
        prompt_emb = embed(prompt_ids)

        # --- CONDITION 1: Inject Embedding of just "8" ---
        # This tests if the model can understand "Here is the answer '8', now say '8'"
        val_token_id = tok(str(target_val), add_special_tokens=False).input_ids[0]
        # Make a memory vector that is EXACTLY the embedding of the digit
        mem_digit = embed(torch.tensor([[val_token_id]], device=args.device)) # (1,1,D)
        
        # Forward pass
        in_digit = torch.cat([mem_digit, prompt_emb], dim=1)
        with torch.no_grad():
            out = model(inputs_embeds=in_digit)
        pred_digit = out.logits[0, -1].argmax().item()
        
        # --- CONDITION 2: Inject Embedding of "KEY01 = 8" ---
        # This tests if the model can read a Key-Value pair from memory
        kv_str = f"{target_key} = {target_val}."
        kv_ids = tok(kv_str, add_special_tokens=False).input_ids
        mem_kv = embed(torch.tensor([kv_ids], device=args.device)).mean(dim=1, keepdim=True) # Mean pool
        
        # Forward pass
        in_kv = torch.cat([mem_kv, prompt_emb], dim=1)
        with torch.no_grad():
            out_kv = model(inputs_embeds=in_kv)
        pred_kv = out_kv.logits[0, -1].argmax().item()

        # Check
        # We need to check if prediction matches target val token
        is_corr_digit = (pred_digit == val_token_id)
        is_corr_kv = (pred_kv == val_token_id)
        
        correct_digit += is_corr_digit
        correct_kv += is_corr_kv
        total += 1
        
        if i < 5:
            print(f"Ex {i}: Target={target_val} | Inj(Digit) Pred={tok.decode([pred_digit])} ({'✅' if is_corr_digit else '❌'}) | Inj(KV) Pred={tok.decode([pred_kv])} ({'✅' if is_corr_kv else '❌'})")

    print(f"\nResults over {total} episodes:")
    print(f"  Accuracy with Perfect Digit Injection: {100*correct_digit/total:.1f}%")
    print(f"  Accuracy with Perfect KV Injection:    {100*correct_kv/total:.1f}%")

    if correct_digit < 50:
        print("\n🚨 CRITICAL FINDING: The model cannot even use the PERFECT embedding.")
        print("   This confirms the Fundamental Architecture Issue.")
        print("   You MUST use the 'Soft Latent' Curriculum approach.")
    else:
        print("\n✅ The model CAN use the embedding! The problem is your Encoder.")

if __name__ == "__main__":
    main()