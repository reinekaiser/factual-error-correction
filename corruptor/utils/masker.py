import random

def mask(src=None, evidence=None, tokenizer=None, mask_ratio=0.15):
    src_tokens = tokenizer.tokenize(src)
    evidence_tokens = tokenizer.tokenize(evidence or "")
    
    lower_src = [t.lower() for t in src_tokens]
    lower_evi = [t.lower() for t in evidence_tokens]
    common = set(lower_src).intersection(lower_evi)
    
    mask_candidates = [i for i, w in enumerate(lower_src) if w not in common]
    if not mask_candidates:
        return tokenizer.convert_tokens_to_string(src_tokens)

    k = max(1, min(int(len(mask_candidates) * mask_ratio), len(mask_candidates)))
    mask_id_list = random.sample(mask_candidates, k=k)

    for mid in mask_id_list:
        src_tokens[mid] = "<mask>"

    return tokenizer.convert_tokens_to_string(src_tokens)

