import torch
from transformers import AutoTokenizer
from src.latent_zip import LatentZipModel

# Mock data
def get_batch(tok, device):
    textA = "KEY01 = 5\nKEY02 = 7\n"
    textB = "Question: what is KEY01?\nAnswer: 5"
    idA = tok(textA, return_tensors="pt").input_ids.to(device)
    idB = tok(textB, return_tensors="pt").input_ids.to(device)
    labels = idB.clone()
    return idA, idB, labels

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda"

print("Loading Model...")
model = LatentZipModel(model_name, num_zip_tokens=8).to(device).to(dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(model_name)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("Running Forward/Backward...")
idA, idB, labels = get_batch(tok, device)
outputs = model(idA, idB)
loss = torch.nn.functional.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
loss.backward()

print(f"Zip Embeds Grad: {model.encoder.zip_embeddings.grad}")
if model.encoder.zip_embeddings.grad is not None:
    print(f"Norm: {model.encoder.zip_embeddings.grad.norm().item()}")
else:
    print("FAIL: Grad is None")
