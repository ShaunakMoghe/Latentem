import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-3B-Instruct"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    load_in_4bit=True,          # start here on 8GB
    trust_remote_code=True,
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "In 2 sentences, explain what a KV cache is in transformers."},
]

prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(model.device)

out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
