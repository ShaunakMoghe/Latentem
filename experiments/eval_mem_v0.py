import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.latent_mem import LatentCompressor

MODEL = "Qwen/Qwen2.5-3B-Instruct"
EVAL_EPISODES = 2000
BATCH = 16


def make_episode(rng: random.Random, max_turns: int):
    keys = []
    for i in range(1, max_turns + 1):
        val = rng.randint(0, 9)
        keys.append((f"KEY{i:02d}", str(val)))

    target_idx = rng.randint(0, len(keys) - 1)
    target_key, target_val = keys[target_idx]

    lines = []
    for (k, v) in keys:
        lines.append(f"{k} VALUE {v}")
    lines.append(f"Question: What is {target_key}? Reply with only the digit.")
    text = "\n".join(lines)

    y = int(target_val)
    return text, y


class EpisodeDS(Dataset):
    def __init__(self, seed: int, n: int, max_turns: int):
        rng = random.Random(seed)
        self.data = [make_episode(rng, max_turns) for _ in range(n)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate(batch, tok, device):
    texts, y = zip(*batch)
    enc = tok(list(texts), return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    y = torch.tensor(y, device=device, dtype=torch.long)
    return input_ids, attn, y


class MemClassifier(nn.Module):
    """
    Query-conditioned readout:
      query (B,D) attends over mem (B,K,D) -> read (B,D) -> classify (B,10)
    """
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.read_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, mem, query_vec):
        # mem: (B,K,D)
        # query_vec: (B,D)
        q = query_vec.unsqueeze(1)                  # (B,1,D)
        read, _ = self.read_attn(q, mem, mem, need_weights=False)  # (B,1,D)
        read = read.squeeze(1)                      # (B,D)
        return self.net(read)


def get_query_vec(last_hidden, attn_mask):
    """
    last_hidden: (B,T,D)
    attn_mask: (B,T) 1 for real tokens, 0 for pad
    Returns: (B,D) hidden state of last non-pad token
    """
    idx = attn_mask.long().sum(dim=1) - 1          # (B,)
    B = last_hidden.shape[0]
    return last_hidden[torch.arange(B, device=last_hidden.device), idx]  # (B,D)


def get_last_hidden(model, input_ids, attn_mask):
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise RuntimeError("Expected model.model to exist for hidden state extraction.")

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            out = backbone(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)
        return out.last_hidden_state.float()


def main():
    ckpt = torch.load("logs/mem_v0.pt", map_location="cpu")

    K = ckpt["K"]
    d_model = ckpt["d_model"]
    bottleneck = ckpt["bottleneck"]
    n_classes = ckpt["n_classes"]
    max_turns = ckpt["max_turns"]
    seed = ckpt["seed"]

    qconf = BitsAndBytesConfig(load_in_4bit=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="auto", quantization_config=qconf, trust_remote_code=True
    )
    model.eval()
    device = model.device

    compressor = LatentCompressor(d_model=d_model, K=K).to(device)
    compressor.load_state_dict({k: v.float() for k, v in ckpt["compressor"].items()})
    compressor.eval()

    head = MemClassifier(d_model=d_model, K=K, bottleneck=bottleneck, n_classes=n_classes).to(device)
    head.load_state_dict({k: v.float() for k, v in ckpt["head"].items()})
    head.eval()

    ds = EpisodeDS(seed + 999, EVAL_EPISODES, max_turns=max_turns)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, collate_fn=lambda b: collate(b, tok, device))

    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attn, y in dl:
            h = get_last_hidden(model, input_ids, attn)
            mem = compressor(h, seq_mask=attn.bool())
            logits = head(mem)
            preds = torch.argmax(logits, dim=-1)
            correct += int((preds == y).sum().item())
            total += y.numel()

    print(f"eval episodes={EVAL_EPISODES} acc={correct/total:.4f} (task=retrieval_digit_with_query, turns={max_turns}, K={K})")


if __name__ == "__main__":
    main()
