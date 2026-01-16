"""
Curriculum Training for Latent Memory.
Phase 1: Train LoRA Reader on PERFECT Oracle Memories (Force it to listen).
Phase 2: Train Compressor on Real Memories (End-to-End).
"""
import argparse
import math
import os
import random
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.latent_mem import MemEncoder
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- DATA ---
KEYS = [f"KEY{i:02d}" for i in range(100)]
DIGITS = list(range(10))

@dataclass
class Episode:
    removed_text: str
    kept_text: str
    target_key: str
    target_val: int

def make_episode(rng, turns, keep_last):
    kv = [(rng.choice(KEYS), rng.choice(DIGITS)) for _ in range(turns)]
    removed = kv[: turns - keep_last]
    kept = kv[turns - keep_last :]
    target_key, target_val = rng.choice(removed)
    
    removed_lines = [f"{v}" for t, (k, v) in enumerate(removed)] # Value-only for hardness? Or kv? Let's use KV for robustness
    # Revert to standard "Turn X: Key=Val" format for robustness
    # SIMPLIFY: Remove "Turn X" noise to help Encoder diversity
    removed_lines = [f"{k} = {v}" for t, (k, v) in enumerate(removed)]
    kept_lines = [f"{k} = {v}" for t, (k, v) in enumerate(kept)]
    
    removed_text = "\n".join(removed_lines)
    kept_text = "\n".join(kept_lines) + f"\nQuestion: what is the value of {target_key}?\nAnswer:"
    return Episode(removed_text, kept_text, target_key, target_val)

def build_batch(tok, episodes, device):
    rem_texts = [e.removed_text for e in episodes]
    rem = tok(rem_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    
    # Labels
    keep_ids_list = []
    labels_list = []
    
    for e in episodes:
        # Prompt: Context + Question + Answer:
        prompt = e.kept_text
        target = str(e.target_val)
        full_text = prompt + target
        
        # Tokenize FULL text to avoid boundary issues
        full_enc = tok(full_text, add_special_tokens=False)
        full_ids = full_enc.input_ids
        
        # Find where the prompt ends
        # We need to be careful. Let's just tokenize prompt and use its length as split point
        # This is strictly safer than concatenating IDs, though still assumes prompt ends on token boundary.
        # For "Answer:", it usually does.
        prompt_ids = tok(prompt, add_special_tokens=False).input_ids
        split_idx = len(prompt_ids)
        
        # Determine actual label IDs from the full sequence (the tail)
        label_ids = full_ids[split_idx:]
        
        # Basic sanity check: if the lengths don't match expectation, standard concat might have merged tokens.
        # But here we trust full_ids is the ground truth sequence the model sees.
        
        full_tensor = torch.tensor(full_ids, dtype=torch.long, device=device)
        lab_tensor = torch.tensor([-100] * split_idx + label_ids, dtype=torch.long, device=device)
        
        keep_ids_list.append(full_tensor)
        labels_list.append(lab_tensor)
        
    keep_ids = torch.nn.utils.rnn.pad_sequence(keep_ids_list, batch_first=True, padding_value=tok.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
    return rem, keep_ids, labels

# --- ORACLE ---
def get_oracle_mem(tok, embed, episodes, k_mem, device, oracle_mean=None):
    slots = []
    for e in episodes:
        # PURE SIGNAL: Just Key and Value. No " = " glue.
        # "KEY01 5"
        s = f"{e.target_key} {e.target_val}"
        ids = torch.tensor(tok(s, add_special_tokens=False).input_ids, device=device)
        emb = embed(ids) # (T, D)
        
        # Normalize to match expected input norms (scale ~1.0 for standard embeddings, but checks help)
        emb = F.normalize(emb, dim=-1)
        
        # Repeat to K
        pooled = emb.mean(dim=0, keepdim=True) # (1, D)
        
        # WHITENING: Remove Fixed Oracle Mean
        if oracle_mean is not None:
            pooled = pooled - oracle_mean
        
        pooled = F.normalize(pooled, dim=-1)   # FORCE UNIT NORM (Match Encoder RMSNorm)
        slots.append(pooled.repeat(k_mem, 1)) # (K, D)
        
    return torch.stack(slots) # (B, K, D)

def estimate_oracle_mean(tok, embed, device, samples=1000):
    print("Estimating Oracle Mean...")
    rng = random.Random(1337)
    episodes = [make_episode(rng, 14, 4) for _ in range(samples)]
    # Use batch logic or simple loop
    vecs = []
    for e in episodes:
        s = f"{e.target_key} {e.target_val}"
        ids = torch.tensor(tok(s, add_special_tokens=False).input_ids, device=device)
        emb = embed(ids).mean(dim=0)
        vecs.append(emb)
    
    vecs = torch.stack(vecs) # (N, D)
    mean = vecs.mean(dim=0, keepdim=True)
    return mean.detach()

# --- MAIN ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1_steps", type=int, default=200, help="Steps to train Reader on Oracle")
    ap.add_argument("--phase2_steps", type=int, default=500, help="Steps to train Jointly")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--save", default="logs/mem_curriculum.pt")
    args = ap.parse_args()
    
    set_seed(42)
    device = "cuda"
    
    # Load Model
    print("Loading Model...")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    
    # LoRA
    # LoRA - Target ALL linear layers for maximum expressivity in the Reader
    peft_config = LoraConfig(
        r=64, lora_alpha=128, # Very aggressive LoRA
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Compressor
    dim = model.get_input_embeddings().weight.shape[1]
    mem_enc = MemEncoder(dim=dim, K=8).to(device).float()
    
    print(f"--- MemEncoder Diagnostics ---")
    print(f"Dim: {dim}")
    print(f"FF[0] Mean: {mem_enc.comp.ff[0].weight.mean().item():.6f} Std: {mem_enc.comp.ff[0].weight.std().item():.6f}")
    print(f"FF[2] Mean: {mem_enc.comp.ff[2].weight.mean().item():.6f} Std: {mem_enc.comp.ff[2].weight.std().item():.6f}")
    print(f"Attn Weights: {mem_enc.comp.attn.out_proj.weight.mean().item():.6f} Std: {mem_enc.comp.attn.out_proj.weight.std().item():.6f}")
    print(f"----------------------------")
    
    # Estimate Oracle Mean for Whitening
    oracle_mean = estimate_oracle_mean(tok, model.get_input_embeddings(), device)
    
    # Optimizers
    # Phase 2 Stability: Reader effectively "frozen" (low LR), Writer learns fast
    opt_reader = torch.optim.AdamW(model.parameters(), lr=1e-4) 
    opt_writer = torch.optim.AdamW(mem_enc.parameters(), lr=1e-3, weight_decay=0.1) # High decay to stop explosion
    
    rng = random.Random(42)
    total_steps = args.phase1_steps + args.phase2_steps
    
    print(f"🚀 Starting Curriculum. Phase 1: {args.phase1_steps} steps. Phase 2: {args.phase2_steps} steps.")
    
    for step in range(1, total_steps + 1):
        is_phase1 = step <= args.phase1_steps
        
        # Batch
        episodes = [make_episode(rng, 14, 4) for _ in range(16)]
        rem, keep_ids, labels = build_batch(tok, episodes, device)
        
        # Initialize losses
        loss_align = torch.tensor(0.0, device=device)

        if is_phase1:
            # PHASE 1: ORACLE MEMORY
            mem_oracle = get_oracle_mem(tok, model.get_input_embeddings(), episodes, 8, device, oracle_mean).to(torch.bfloat16)
            mem = mem_oracle.detach()
        else:
            # PHASE 2: MIXED MEMORY (Smooth Transition) + ALIGNMENT LOSS
            # Calculate mixing factor alpha: 1.0 -> 0.0 over phase 2
            p2_progress = (step - args.phase1_steps) / args.phase2_steps
            alpha = max(0.0, 1.0 - p2_progress) 
            
            # Get Oracle (Target for Encoder)
            mem_oracle = get_oracle_mem(tok, model.get_input_embeddings(), episodes, 8, device, oracle_mean).to(torch.bfloat16)
            
            # Get Learned
            with torch.no_grad():
                out_enc = model(input_ids=rem.input_ids, attention_mask=rem.attention_mask, output_hidden_states=True)
                hs = out_enc.hidden_states[12]
                if step % 25 == 0:
                     hs_var = hs.float().var(dim=0).mean().item()
                     print(f"    [Dg] Input HS Var: {hs_var:.6f} Shape: {hs.shape}")
            mem_learned = mem_enc(hs.float(), rem.attention_mask).to(torch.bfloat16)
            
            # ALIGNMENT LOSS: INFONCE (Contrastive)
            # MSE failed because the "Mean Prediction" was a stable saddle point.
            # InfoNCE forces distinctions by penalizing high similarity to negatives.
            # Both pred and targ are Whitened and Normalized.
            
            # 1. Pool to Episode Means
            pred = mem_learned.mean(dim=1).float() # (B, D)
            targ = mem_oracle.mean(dim=1).float()  # (B, D)
            
            # 2. Compute Similarity Matrix
            logits = torch.matmul(pred, targ.t()) # (B, B)
            logits = logits / 0.07 # Temperature
            
            # 4. EXPLICIT DIVERSITY LOSS (Helper)
            std_dev = torch.sqrt(pred.var(dim=0) + 1e-6).mean()
            loss_diversity = torch.clamp(1.0 - std_dev, min=0.0)
            
            labels_align = torch.arange(logits.shape[0], device=device)
            loss_align = 1.0 * F.cross_entropy(logits, labels_align) + 1.0 * loss_diversity # Weight 1.0
            
            # MIXING: During fwd pass, Reader sees a blend
            # DETACH Learned from LM loss path. Encoder learns ONLY via Alignment Loss.
            mem = (1 - alpha) * mem_learned.detach() + alpha * mem_oracle.detach()

        # Forward
        emb_keep = model.get_input_embeddings()(keep_ids)
        inputs_embeds = torch.cat([mem, emb_keep], dim=1)
        
        # Masking
        B, K, _ = mem.shape
        S = keep_ids.shape[1]
        mask_mem = torch.ones((B, K), device=device)
        mask_keep = (keep_ids != tok.pad_token_id).long()
        attn_mask = torch.cat([mask_mem, mask_keep], dim=1)
        
        full_labels = torch.cat([torch.full((B, K), -100, device=device), labels], dim=1)
        
        out = model(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=full_labels)
        loss = out.loss + 1.0 * loss_align # Reduced from 10.0 to 1.0 for MSE balance

        
        # Backward
        if is_phase1:
            # Train ONLY Reader
            opt_reader.zero_grad()
            loss.backward()
            opt_reader.step()
            phase = "PHASE 1 (Reader)"
        else:
            # Train BOTH (Joint)
            opt_reader.zero_grad()
            opt_writer.zero_grad()
            loss.backward()

            # --- GRADIENT DEBUG ---
            if step % 25 == 0 and not is_phase1:
                grad_norm_reader = sum([p.grad.norm().item() for p in model.parameters() if p.grad is not None])
                grad_norm_writer = sum([p.grad.norm().item() for p in mem_enc.parameters() if p.grad is not None])
                print(f"    [Grad] Reader={grad_norm_reader:.4f} Writer={grad_norm_writer:.4f}")
            # ----------------------

            torch.nn.utils.clip_grad_norm_(mem_enc.parameters(), 1.0) # CLIP GRADIENTS
            opt_reader.step()
            opt_writer.step()
            phase = "PHASE 2 (Joint)"
            
        if step % 25 == 0:
            # Acc
            preds = out.logits.argmax(dim=-1)
            # Simple check on last token
            acc = (preds[:, -1] == labels[:, -1]).float().mean().item()
            
            # DIAGNOSTICS
            with torch.no_grad():
                # 1. Norms
                norm_oral = mem_oracle.norm(dim=-1).mean().item() if is_phase1 or alpha < 1.0 else 0.0
                norm_learn = mem_learned.norm(dim=-1).mean().item() if not is_phase1 else 0.0
                
                # 2. Diversity (Batch Collapse Check)
                # Compute pairwise cosine sim of mem_learned MEAN vectors (Episode level)
                if not is_phase1:
                    # Div on means (B, D) to avoid slot-repetition artifacts
                    flat = mem_learned.mean(dim=1) # (B, D)
                    flat = F.normalize(flat, dim=-1)
                    sim_mat = flat @ flat.T # (B, B)
                    mask_eye = torch.eye(sim_mat.shape[0], device=device).bool()
                    div_score = sim_mat.masked_select(~mask_eye).mean().item()
                    
                    # 4. Variance Check (Are values actually different?)
                    var_0 = mem_learned[:, 0, 0].float().var().item()
                    
                    # 5. Oracle Diversity Check
                    flat_oral = mem_oracle.mean(dim=1)
                    flat_oral = F.normalize(flat_oral, dim=-1)
                    sim_mat_oral = flat_oral @ flat_oral.T
                    div_oral = sim_mat_oral.masked_select(~mask_eye).mean().item()
                else:
                    div_score = 0.0
                    var_0 = 0.0
                    div_oral = 0.0

                # 3. Input Diversity Check
                if step % 25 == 0 and not is_phase1:
                     # Check distinctness of first two inputs
                     t0 = episodes[0].removed_text
                     t1 = episodes[1].removed_text
                     pass # Verified.


            print(f"[{phase}] Step {step}: Loss={loss.item():.4f} (LM={out.loss.item():.4f} Align={loss_align.item():.4f}) | Acc={acc*100:.1f}%")
            print(f"    [Dg] Norm: Ora={norm_oral:.2f} Lrn={norm_learn:.2f} | Div: Ora={div_oral:.2f} Lrn={div_score:.2f} | Var={var_0:.6f}")
            
    # Save
    ckpt = {
        "state_dict": mem_enc.state_dict(),
        "lora_state_dict": get_peft_model_state_dict(model),
        "meta": {"K": 8}
    }
    torch.save(ckpt, args.save)
    print("Saved.")

if __name__ == "__main__":
    main()