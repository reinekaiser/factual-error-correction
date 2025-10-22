import random
import re
import math

import random

def mask(src=None, evidence=None, tokenizer=None, mask_ratio=0.15):
    """
    ViT5-style span masking (safe version)
    - Che ngẫu nhiên một số span trong `src`
    - Dùng sentinel tokens <extra_id_0>, <extra_id_1>, ...
    - Hợp với tokenizer kiểu SentencePiece (T5/mT5/Flan-T5)
    """
    if not src or tokenizer is None:
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

    num_spans = max(1, k // 3)
    span_starts = sorted(random.sample(mask_candidates, num_spans))

    masked_tokens = []
    current_id = 0
    i = 0
    n = len(src_tokens)
    used_starts = set(span_starts)

    while i < n:
        if i in used_starts:
            span_len = random.randint(1, 3)
            masked_tokens.append(f" *")
            current_id += 1
            i = min(i + span_len, n)
        else:
            masked_tokens.append(src_tokens[i])
            i += 1

        if len(masked_tokens) > n * 3:
            print("Loop protection triggered")
            break

    masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
    return masked_text
