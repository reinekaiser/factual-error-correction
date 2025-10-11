import random
import inspect
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

class Masker:
    def __init__(self, tokenizer_name="xlm-roberta-base", device=-1, mask_prob = 0.15):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mask_prob = mask_prob
        self.fill_pipeline = pipeline(
            "fill-mask",
            model=tokenizer_name,
            tokenizer=tokenizer_name,
            device=device
        )
        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

    def tokenizer_mask_and_fill_wrong(self, claim, context="", top_k=10):
        """
        1. Mask 1-2 span (chunk) trong claim
        2. Fill [MASK] bằng MLM nhưng tránh token xuất hiện trong context/evidence
        Trả về:
            masked_text: câu có <mask>
            filled_text: câu plausibly wrong, fluent
        """
        # ----- Step 1: mask span -----
        words = claim.split()
        masked_positions = []

        # Tách chunks noun/verb
        try:
            tags = pos_tag(claim)
            chunks = chunk(tags)
            candidate_chunks = [c for c in chunks if c['type'] in ('NP','VP')]
            if candidate_chunks:
                n_span = min(len(candidate_chunks), random.randint(1,2))
                selected_chunks = random.sample(candidate_chunks, n_span)
                for c in selected_chunks:
                    c_words = c['text'].split()
                    try:
                        start = next(i for i in range(len(words)) if words[i:i+len(c_words)] == c_words)
                        masked_positions.extend(range(start, start+len(c_words)))
                    except StopIteration:
                        continue
        except Exception:
            pass

        # fallback: mask random token
        if not masked_positions:
            masked_positions = [i for i in range(len(words)) if random.random() < self.mask_prob]

        # Mask tokens
        for i in masked_positions:
            words[i] = self.tokenizer.mask_token

        masked_text = " ".join(words)

        # ----- Step 2: MLM fill -----
        filled_text = masked_text
        if self.tokenizer.mask_token in masked_text:
            filled_candidates = self.fill_pipeline(masked_text, top_k=top_k)
            context_tokens = set(context.split()) if context else set()
            for c in filled_candidates:
                candidate_seq = c['sequence']
                if not any(tok in context_tokens for tok in candidate_seq.split()):
                    filled_text = candidate_seq
                    break
            if not filled_text:
                filled_text = masked_text

        return masked_text, filled_text


