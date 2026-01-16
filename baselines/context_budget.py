import re
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Block:
    kind: str
    text: str

COMPRESSIBLE_KINDS = {"SCRATCH", "THOUGHT", "SUMMARY"}

def render_blocks(blocks: List[Block]) -> str:
    return "\n".join([f"[{b.kind}]\n{b.text}" for b in blocks])

def count_tokens(tokenizer, blocks: List[Block]) -> int:
    return len(tokenizer(render_blocks(blocks), add_special_tokens=False).input_ids)

def apply_discard(tokenizer, blocks: List[Block], max_tokens: int) -> List[Block]:
    new_blocks = list(blocks)
    while count_tokens(tokenizer, new_blocks) > max_tokens:
        idx = next((i for i,b in enumerate(new_blocks) if b.kind in COMPRESSIBLE_KINDS), None)
        if idx is None:
            break
        new_blocks.pop(idx)
    return new_blocks

_KEY_RE = re.compile(r"\b(KEY\d{2})\s*=\s*([A-Z0-9\-]+)\b")

def apply_extractive_kv_summary(tokenizer, blocks: List[Block], max_tokens: int) -> List[Block]:
    """
    Strong text baseline: deterministically extract KEYxx=VALUE pairs and keep them in a compact table.
    This avoids LLM hallucination/omission in the summary baseline.
    """
    if count_tokens(tokenizer, blocks) <= max_tokens:
        return blocks

    keep_blocks = [b for b in blocks if b.kind == "KEEP"]
    compress_blocks = [b for b in blocks if b.kind in COMPRESSIBLE_KINDS]

    # Extract pairs in chronological order of appearance (oldest first)
    pairs: List[Tuple[str,str]] = []
    seen = set()
    for b in compress_blocks:
        for k, v in _KEY_RE.findall(b.text):
            if k not in seen:
                seen.add(k)
                pairs.append((k, v))

    # Build canonical summary text
    lines = [f"{k} = {v}" for k, v in pairs]
    summary_text = "\n".join(lines) if lines else ""

    # Candidate with one summary block
    candidate = keep_blocks + [Block("SUMMARY", summary_text)]
    if count_tokens(tokenizer, candidate) <= max_tokens:
        return candidate

    # If still too big, drop *newest* keys until it fits (keeps KEY01 as long as possible)
    while lines and count_tokens(tokenizer, keep_blocks + [Block("SUMMARY", "\n".join(lines))]) > max_tokens:
        lines.pop()  # drop last (newest)
    candidate = keep_blocks + [Block("SUMMARY", "\n".join(lines))]
    if count_tokens(tokenizer, candidate) <= max_tokens:
        return candidate

    # If even KEEP alone too big, last resort: discard
    return apply_discard(tokenizer, blocks, max_tokens)
