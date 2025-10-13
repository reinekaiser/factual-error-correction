import random
import re

def mask(src=None, evidence=None, tokenizer=None, mask_ratio=0.15):
    """
    ViT5-style span masking:
    - Che một phần text không xuất hiện trong evidence
    - Dùng sentinel tokens <extra_id_0>, <extra_id_1>, ...
    - Hợp với tokenizer kiểu SentencePiece (ViT5/mT5/T5)
    """
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

    k = int(len(mask_candidates) * mask_ratio)
    k = max(1, min(k, len(mask_candidates)))
    
    # --- Chia thành các span ---
    num_spans = max(1, k // 3)  # ví dụ: mỗi span dài trung bình ~3 token
    span_starts = sorted(random.sample(mask_candidates, num_spans))
    
    masked_tokens = []
    current_id = 0
    i = 0
    while i < len(src_tokens):
        if i in span_starts:
            span_len = random.randint(1, 3)
            masked_tokens.append(f"<extra_id_{current_id}>")
            i += span_len
            current_id += 1
        else:
            masked_tokens.append(src_tokens[i])
            i += 1

    masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
    return masked_text


