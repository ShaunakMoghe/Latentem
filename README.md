# Research Documentation

## 1) Project Title

**Latentem: Latent Memory Compression for Long-Context Retrieval in Decoder-Only LLMs**

## 2) Core Goal and Motivation

The goal of Latentem was to build and validate a **learned latent memory module** that can compress “removed” context into a small set of continuous vectors (“memory slots”), then **inject** those vectors back into a decoder-only transformer (Qwen/Qwen2.5-0.5B-Instruct) so the model can still answer questions that require information from the removed portion of the prompt.

Motivation:

* Long contexts are expensive (KV cache growth, compute, memory).
* Token dropping or truncation loses essential facts.
* A small learned memory could preserve key information more efficiently than keeping all tokens.

High-level hypothesis:

* A small trainable compressor can encode lost context into K latent vectors.
* Injecting those latents into Qwen will recover answer accuracy under constrained budgets.

## 3) Experimental Setup (What Was Built)

### Base LLM

* **Qwen/Qwen2.5-0.5B-Instruct**, run locally on consumer GPU.
* Model weights remained **frozen** throughout most experiments.

### Memory Module

* Implemented in `src/latent_mem.py` (MemEncoder / LatentCompressor).
* Produced **K memory vectors** (e.g., K=8).
* Memory vectors were injected during evaluation to replace/compensate for removed text.

### Training Script

* Primary training driver: `experiments/train_compressor_v1_digitfirst.py` (and earlier variants).
* Multiple debugging modes were added (conditioning check, gradient check, parameter update check, overfit-one).

### Evaluation Script

* `experiments/eval_budget_sweep_v0.py`
* Swept context budgets (e.g., 64/96/128 tokens), measured accuracy under:

  * **discard baseline** (removed info lost),
  * **summary baseline** (explicitly included summary tokens),
  * **latent** (latent memory injection).

### Synthetic Task: “Digit-first” KV recall

A controlled synthetic task was used to isolate memory effects:

* Prompt contains multiple “turns” formatted as structured KV statements (e.g., `KEY05 = 8`).
* A target key’s value (a single digit 0–9) is placed in the **removed** segment.
* The model must answer the digit after truncation.
* Summary baseline was expected to approach 100% when budget allowed including the relevant KV/digit.

## 4) Key Intended Success Criteria

1. **Latent accuracy significantly above random** on digit recall when the relevant KV digit is removed.
2. **Latent accuracy improves with budget** (as injection has more room to help).
3. Learned memory is **input-dependent** (not constant across episodes).
4. Oracle sanity checks indicate the **injection pathway works** when correct memory is provided.

## 5) Major Observations and Results

### A) Summary baseline worked (task is solvable)

Across runs, the **summary baseline** achieved very high accuracy at larger budgets (often near ~98–100%), indicating:

* The underlying model can answer the task if the relevant information is present.
* The evaluation pipeline for “kept context → answer” was functioning.

### B) Latent memory failed to improve accuracy

Learned latent memory injection produced accuracy near random guessing (~6–10%), showing:

* The compressor was not providing useful digit information.
* Improvements were not stable across budgets or seeds.

### C) Oracle confirmed injection path is correct

When using **oracle memory** that directly contained the relevant information (e.g., `gold_kv_seq`), accuracy reached **100%**.
This was a critical validation:

* The model can use injected memory if it contains the right signal.
* The failure is primarily in **learning/generating correct memory**, not in the downstream injection mechanism.

### D) MemEncoder collapse (nearly constant outputs)

A repeated and decisive debug signal was:

* **cosine(mem_episode_t, mem_episode_t-1) ≈ 1.0**
* Conditioning checks showed **very high cosine similarity across batch elements** (~0.996–0.999).

Interpretation:

* The memory module was producing nearly the same output regardless of input.
* This indicates **representation collapse** / weak conditioning / failure to encode distinct digits.

### E) Overfit-one could fit a single episode but didn’t generalize

The model could drive training loss near zero and reach high accuracy when repeating one episode (“overfit-one”), but evaluation remained poor.
Interpretation:

* The system can memorize a single mapping under extremely favorable conditions.
* It fails to learn a robust input-dependent compressor that generalizes.

## 6) Why Some “Progress” Was Misleading (False Positive)

At one point, training appeared to improve when the “oracle target” was built from the **full KV string** (e.g., `KEY07 = 5.`) or KV sequence and compared via cosine similarity.
This was misleading because:

* The data uses a rigid shared template (“KEY”, “=”, punctuation, formatting tokens) across examples.
* Any pooled embedding of the full KV string is dominated by these shared tokens.
* The compressor can reduce oracle loss by matching the **template embedding** rather than extracting the digit.

When the oracle target was changed to **gold_digit** (digit only), the model could no longer “cheat,” and training did not improve—revealing the true failure mode.

## 7) Final Diagnosis: Why It Didn’t Work

### Primary Failure Mode: Frozen-reader mismatch + weak learning signal

The system required the frozen Qwen model to “read” new latent vectors whose distribution it was never trained on. With Qwen frozen:

* There is no adaptive mechanism for Qwen’s attention/MLP layers to interpret the new memory format.
* Training relies entirely on gradients shaping the compressor so its output “looks like” something Qwen can use.
* On this task, that signal was too weak and unstable for digit-level information.

### Secondary Failure Mode: Template-dominated conditioning on highly repetitive data

The synthetic dataset is structurally homogeneous:

* Most tokens are identical across samples (template).
* The digit signal is sparse.
  Mean pooling / simple conditioning mechanisms tend to wash out small differences, encouraging collapse early.

### Empirical Evidence Supporting “Signal Failure”

* Persistent near-constant memory outputs (cosine ~1.0).
* Latent accuracy near random despite extensive training.
* Oracle injection achieves 100% (so “if memory were correct, the system would work”).
* “Template oracle” improves but “digit-only oracle” does not (model learns background but not content).

## 8) Conclusion

Latentem successfully validated the evaluation harness, the notion of budgeted context removal, and the memory injection pathway via oracle memory. However, the core objective—**training a latent compressor that reliably encodes removed information such that a frozen decoder-only LLM can use it**—did not succeed.

The dominant conclusion is that the **frozen-reader + learned latent writer** setup was not able to propagate a sufficiently strong learning signal to encode fine-grained content (digits) under this architecture and dataset. The learned compressor collapsed toward nearly constant outputs and did not produce informative memory vectors, resulting in latent performance near random guessing even when summary baselines and oracle memory showed the task was otherwise solvable.

## 9) Lessons Learned

* Oracle memory sanity checks are essential and were decisive in isolating failure source.
* Synthetic tasks can accidentally allow “shortcut learning” (template matching) that looks like progress but doesn’t reflect true capability.
* Latent-vector injection likely requires a trainable reader (e.g., LoRA / adapter layers) or a different interface (e.g., token-based retrieval/selection) to avoid “frozen model blindness.”

## 10) Suggested Next Directions (if revisited in the future)

If this line of work is revisited, the most promising pivots are:

1. **Trainable reader via LoRA** so the LLM learns to interpret memory vectors.
2. **Token selection / sparse retrieval** instead of latent compression, leveraging the model’s native token interface.
3. **Contrastive or supervised objectives that explicitly target digit-level discriminability** and penalize collapse.
4. **More diverse data / less templated tasks** to prevent pooling from washing out the information of interest.

## 11) Artifacts / Files Referenced

* `src/latent_mem.py` — memory module implementation (MemEncoder / LatentCompressor)
* `experiments/train_compressor_v1_digitfirst.py` — training driver with debug modes
* `experiments/eval_budget_sweep_v0.py` — evaluation script with budget sweeps and oracle tests
* Checkpoints: `logs/mem_encoder_*` (multiple variants, including overfit-one and normalized experiments)
