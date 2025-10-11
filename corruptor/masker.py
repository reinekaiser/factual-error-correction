import random
import inspect
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

class Masker:
    def __init__(self, tokenizer_name="xlm-roberta-base", device=-1, mask_prob = 0.15):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mask_prob = mask_prob

    def tokenizer_mask_and_fill_wrong(self, claim, context="", top_k=10):
        """
        1. Tạo masked_text: claim có token bị [MASK]
        2. Tạo filled_text: MLM fill [MASK], tránh token xuất hiện trong context
        """
        # ----- Step 1: Mask -----
        encoded = self.tokenizer(claim, return_tensors="pt")
        input_ids = encoded["input_ids"][0].clone()
        special_ids = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id}

        for i, tid in enumerate(input_ids):
            if int(tid) in special_ids:
                continue
            if random.random() < self.mask_prob:
                input_ids[i] = self.tokenizer.mask_token_id

        masked_text = self.tokenizer.decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        filled_text = ""
        if "[MASK]" in masked_text:
            filled_candidates = self.fill_pipeline(masked_text, top_k=top_k)
            context_tokens = set(context.split())
            for c in filled_candidates:
                candidate_seq = c['sequence']
                if not any(tok in context_tokens for tok in candidate_seq.split()):
                    filled_text = candidate_seq
                    break
            if not filled_text:
                filled_text = masked_text

        return masked_text, filled_text
