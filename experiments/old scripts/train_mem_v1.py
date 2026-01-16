import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.latent_mem import LatentCompressor

MODEL = "Qwen/Qwen2.5-3B-Instruct"  # Switch to larger model for better representations
SEED = 7

# Memory slots - reduce for simpler task
K_MEM = 8

# Curriculum task: retrieve a single digit value (0..9)
N_CLASSES = 10

# Data / training - increase complexity and data
MAX_TURNS = 10
EPISODES = 10000
BATCH = 32
EPOCHS = 20
LR = 3e-4  # Lower LR for more stable training

# Head bottleneck (keeps checkpoint small, trains fast)
BOTTLENECK = 512


def make_episode(rng: random.Random):
    """
    Generate a log of KEYxx = digit (0..9) pairs, then prepend the query:
    "Question: What is KEYyy?"
    Target is the digit value for KEYyy (10-class classification).
    This order ensures context representations are query-conditioned in causal LM.
    """
    keys = []
    for i in range(1, MAX_TURNS + 1):
        val = rng.randint(0, 9)
        keys.append((f"KEY{i:02d}", str(val)))

    target_idx = rng.randint(0, len(keys) - 1)
    target_key, target_val = keys[target_idx]

    query_line = f"Question: What is {target_key}? Reply with only the digit."

    context_lines = []
    for k, v in keys:
        context_lines.append(f"{k} VALUE {v}")

    text = query_line + "\n" + "\n".join(context_lines)

    y = int(target_val)  # 0..9
    return query_line, text, y


class EpisodeDS(Dataset):
    def __init__(self, seed: int, n: int):
        rng = random.Random(seed)
        self.data = [make_episode(rng) for _ in range(n)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate(batch, tok, device):
    query_lines, texts, y = zip(*batch)
    # Apply chat template for full text (query + context)
    messages = [
        [
            {"role": "system", "content": "You are a helpful assistant that retrieves values accurately."},
            {"role": "user", "content": text}
        ] for text in texts
    ]
    formatted_texts = [tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    enc = tok(formatted_texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    # Tokenize query alone to get lengths
    query_messages = [
        [
            {"role": "system", "content": "You are a helpful assistant that retrieves values accurately."},
            {"role": "user", "content": query_line}
        ] for query_line in query_lines
    ]
    query_formatted = [tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in query_messages]
    query_enc = tok(query_formatted, return_tensors="pt", padding=True, truncation=True)
    query_lengths = query_enc["attention_mask"].sum(dim=1).to(device)  # (B,)

    y = torch.tensor(y, device=device, dtype=torch.long)
    return input_ids, attn, query_lengths, y


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


def get_last_hidden(model, input_ids, attn_mask):
    """
    Extract last_hidden_state from frozen model backbone.
    Uses autocast for speed, returns fp32 for stable training.
    """
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise RuntimeError("Expected model.model to exist for hidden state extraction.")

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            out = backbone(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)
        return out.last_hidden_state.float()  # (B,T,D) fp32


def main():
    os.makedirs("logs", exist_ok=True)
    torch.manual_seed(SEED)
    random.seed(SEED)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="auto", dtype=torch.float16, trust_remote_code=True
    )
    model.eval()

    device = model.device
    d_model = model.config.hidden_size

    compressor = LatentCompressor(d_model=d_model, K=K_MEM).to(device)  # train fp32
    head = MemClassifier(d_model=d_model, n_classes=N_CLASSES).to(device)

    # freeze LLM
    for p in model.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(list(compressor.parameters()) + list(head.parameters()), lr=LR, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    ds = EpisodeDS(SEED, EPISODES)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=lambda b: collate(b, tok, device))

    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0

        for input_ids, attn, query_lengths, y in dl:
            h = get_last_hidden(model, input_ids, attn)           # (B,T,D)

            # Create mask to compress only context part (after query)
            B, T = attn.shape
            context_mask = attn.clone()
            for i in range(B):
                context_mask[i, :query_lengths[i]] = 0

            mem = compressor(h, seq_mask=context_mask.bool())  # Compress context, conditioned on query

            # Query vec: average of query hidden states
            qvecs = []
            for i in range(B):
                q_slice = h[i, :query_lengths[i], :]
                qvecs.append(q_slice.mean(dim=0))
            qvec = torch.stack(qvecs)  # (B, D)

            logits = head(mem, qvec)

            loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(compressor.parameters()) + list(head.parameters()), 1.0)
            opt.step()

            total_loss += float(loss.item())
            preds = torch.argmax(logits, dim=-1)
            correct += int((preds == y).sum().item())
            total += y.numel()

        print(f"epoch {epoch+1}/{EPOCHS} loss={total_loss/len(dl):.4f} acc={correct/total:.4f}")

    # Save smaller checkpoint (fp16 on CPU)
    compressor_fp16 = {k: v.detach().half().cpu() for k, v in compressor.state_dict().items()}
    head_fp16 = {k: v.detach().half().cpu() for k, v in head.state_dict().items()}

    torch.save(
        {
            "compressor": compressor_fp16,
            "head": head_fp16,
            "K": K_MEM,
            "d_model": d_model,
            "bottleneck": BOTTLENECK,
            "n_classes": N_CLASSES,
            "max_turns": MAX_TURNS,
            "episodes": EPISODES,
            "epochs": EPOCHS,
            "seed": SEED,
            "task": "retrieval_digit_with_query",
        },
        "logs/mem_v0.pt",
    )
    print("saved to logs/mem_v0.pt")


if __name__ == "__main__":
    main()