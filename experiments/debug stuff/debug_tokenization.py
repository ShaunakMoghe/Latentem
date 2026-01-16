from transformers import AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)

print(f"Vocab size: {tok.vocab_size}")

digits = range(10)
for d in digits:
    s_val = str(d)
    
    # 1. Tokenize just the digit
    digit_ids = tok(s_val, add_special_tokens=False)["input_ids"]
    
    # 2. Tokenize context + digit 
    context = "Answer:"
    full_ids = tok(context + s_val, add_special_tokens=False)["input_ids"]
    context_ids = tok(context, add_special_tokens=False)["input_ids"]
    
    # Extract the "added" ID
    suffix_ids = full_ids[len(context_ids):]
    
    print(f"Digit: '{d}'")
    print(f"  Alone IDs: {digit_ids}")
    print(f"  Context: '{context}' ({context_ids})")
    print(f"  Full IDs: {full_ids}")
    print(f"  Suffix IDs: {suffix_ids}")
    
    match = (digit_ids == suffix_ids)
    print(f"  Match? {match}")
    if not match:
        print("  ❌ MISMATCH DETECTED!")
