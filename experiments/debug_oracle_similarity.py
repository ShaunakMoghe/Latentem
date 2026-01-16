
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"Loading {model_id}...")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    embed = model.get_input_embeddings()
    
    # Generate some distinct keys
    keys = [f"KEY{i:02d}" for i in range(16)]
    vals = [str(i) for i in range(16)]
    
    print("\n--- Testing Raw Embeddings ---")
    vecs = []
    for k, v in zip(keys, vals):
        # Format used in training
        s = f"{k} {v}"
        ids = torch.tensor(tok(s, add_special_tokens=False).input_ids, device=device)
        # Pool (mean)
        e = embed(ids).mean(dim=0)
        vecs.append(e)
        
    vecs = torch.stack(vecs) # (16, D)
    vecs = F.normalize(vecs, dim=-1)
    
    # Calc Similarity
    sim_mat = vecs @ vecs.T
    # Mask diagonal
    mask = torch.eye(16, device=device).bool()
    avg_sim = sim_mat.masked_select(~mask).mean().item()
    print(f"Average Pairwise Cosine Similarity: {avg_sim:.4f}")
    
    print("\n--- Testing Centered (Whitened) Embeddings ---")
    # Centering: Subtract the mean vector of the batch (or universe)
    # This removes the "common direction" (DC component)
    
    vecs_raw = []
    for k, v in zip(keys, vals):
        s = f"{k} {v}"
        ids = torch.tensor(tok(s, add_special_tokens=False).input_ids, device=device)
        e = embed(ids).mean(dim=0)
        vecs_raw.append(e)
    
    vecs_raw = torch.stack(vecs_raw)
    
    # CENTER
    mean_vec = vecs_raw.mean(dim=0, keepdim=True)
    vecs_centered = vecs_raw - mean_vec
    
    # Normalize AFTER centering
    vecs_centered = F.normalize(vecs_centered, dim=-1)
    
    sim_mat_c = vecs_centered @ vecs_centered.T
    avg_sim_c = sim_mat_c.masked_select(~mask).mean().item()
    print(f"Average Pairwise Cosine Similarity (Centered): {avg_sim_c:.4f}")
    
    if avg_sim_c < 0.5:
        print("\nSUCCESS: Centering reduces similarity significantly.")
    print("\n--- Testing Global Mean Centered Embeddings ---")
    # Global Mean of all tokens
    global_mean = embed.weight.mean(dim=0, keepdim=True)
    
    vecs_global = []
    for k, v in zip(keys, vals):
        s = f"{k} {v}"
        ids = torch.tensor(tok(s, add_special_tokens=False).input_ids, device=device)
        e = embed(ids).mean(dim=0)
        vecs_global.append(e)
    
    vecs_global = torch.stack(vecs_global)
    vecs_global = vecs_global - global_mean
    vecs_global = F.normalize(vecs_global, dim=-1)
    
    sim_mat_g = vecs_global @ vecs_global.T
    avg_sim_g = sim_mat_g.masked_select(~mask).mean().item()
    print(f"Average Pairwise Cosine Similarity (Global Centered): {avg_sim_g:.4f}")

if __name__ == "__main__":
    main()
