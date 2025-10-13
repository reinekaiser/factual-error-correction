import random
import re

def mask(src=None, evidence=None, tokenizer=None, mask_ratio=0.3):
    if not src:
        return ""

    src_tokens = tokenizer.tokenize(src)
    evidence_tokens = tokenizer.tokenize(evidence or "")
    
    lower_src = [t.lower() for t in src_tokens]
    lower_evi = [t.lower() for t in evidence_tokens]
    common = set(lower_src).intersection(lower_evi)

    mask_candidates = [i for i, w in enumerate(lower_src) if w not in common]
    if not mask_candidates:
        return tokenizer.convert_tokens_to_string(src_tokens)

    k = max(1, min(int(len(mask_candidates) * mask_ratio), len(mask_candidates)))
    k = min(k, len(mask_candidates))
    random.shuffle(mask_candidates)

    spans = []
    current = []
    for idx in mask_candidates[:k]:
        if not current or idx == current[-1] + 1:
            current.append(idx)
        else:
            spans.append(current)
            current = [idx]
    if current:
        spans.append(current)

    spans = spans[:5]
    
    masked_tokens = src_tokens.copy()
    sentinel_id = 0

    for span in spans:
        start, end = span[0], span[-1] + 1
        masked_tokens[start:end] = [f"<extra_id_{sentinel_id}>"]
        sentinel_id += 1

    text = tokenizer.convert_tokens_to_string(masked_tokens)
    text = re.sub(r'(<extra_id_\d+>)(\s*\1)+', r'\1', text)

    return text