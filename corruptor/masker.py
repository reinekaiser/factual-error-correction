import random
import inspect
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

class Masker:
    def __init__(self, tokenizer_name="xlm-roberta-base", device=-1, mask_prob = 0.15):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mask_prob = mask_prob

    def tokenizer_mask_and_fill_wrong(self, claim, context="", top_k=10):
        encoded = self.tokenizer(claim, return_tensors="pt")
        input_ids = encoded["input_ids"][0].clone()
        special_ids = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id}
        mask_token_id = self.tokenizer.mask_token_id
        mask_token = self.tokenizer.mask_token  # <mask>

        # Mask random tokens
        for i, tid in enumerate(input_ids):
            if int(tid) in special_ids:
                continue
            if random.random() < self.mask_prob:
                input_ids[i] = mask_token_id

        # Decode giữ mask token
        masked_text = self.tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)

        # Fill
        filled_text = masked_text
        if mask_token in masked_text:
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

