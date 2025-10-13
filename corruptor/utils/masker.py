import random

import random
import re

import random, re

def mask(src=None, evidence = None,tokenizer=None, mask_ratio=0.3):
    """
    Randomly masks a portion of the source text for T5/ViT5-style training.
    - Works with SentencePiece tokenizers (ViT5, mT5, T5, etc.)
    - Uses <extra_id_*> sentinel tokens for span corruption
    """
    if not src:
        return ""

    # Tokenize input
    src_tokens = tokenizer.tokenize(src)

    # Không mask nếu quá ngắn
    if len(src_tokens) < 5:
        return tokenizer.convert_tokens_to_string(src_tokens)

    # Chọn ngẫu nhiên token để mask
    num_to_mask = max(1, int(len(src_tokens) * mask_ratio))
    mask_indices = sorted(random.sample(range(len(src_tokens)), num_to_mask))

    # Gom các token liền kề thành span
    spans = []
    current = [mask_indices[0]]
    for idx in mask_indices[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            spans.append(current)
            current = [idx]
    spans.append(current)

    # Giới hạn số span
    spans = spans[:5]

    masked_tokens = src_tokens.copy()
    sentinel_id = 0

    # Thay từng span bằng <extra_id_n>
    for span in spans:
        start, end = span[0], span[-1] + 1
        masked_tokens[start:end] = [f"<extra_id_{sentinel_id}>"]
        sentinel_id += 1

    # Chuyển lại thành text
    text = tokenizer.convert_tokens_to_string(masked_tokens)

    # Xử lý trường hợp mask lặp
    text = re.sub(r'(<extra_id_\d+>)(\s*\1)+', r'\1', text)

    return text


