import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from underthesea import word_tokenize
import asyncio
import random, math

class Seq2SeqPredictor:
    def __init__(self, model_name_or_path: str, device=None):
        """
        Init: load model + tokenizer, fixed hyperparams
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {model_name_or_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(self.device)
        print("Model and tokenizer loaded.")

        self.max_src_len = 512
        self.max_tgt_len = 128
        self.num_beams = 5
        self.do_sample = False
        self.top_k = 0
        self.top_p = 0.9
        self.num_return_sequences = 1
        self.repetition_penalty = 1.2
        self.temperature = 1.0

    def word_segment(self, text):
        return word_tokenize(text, format="text")

    def mask(self, src_text, evidence=None, mask_token="*", mask_ratio=0.15, mask_strategy='heuristic', merge_mask=False):
        src_tokens = word_tokenize(src_text, format="text").split()
        evidence_tokens = word_tokenize(evidence, format="text").split() if evidence else []

        if mask_strategy == "random":
            src_len = len(src_tokens)
            mask_id_list = random.sample(range(src_len), k=max(1, min(src_len, math.ceil(src_len * mask_ratio))))
            for mask_id in mask_id_list:
                src_tokens[mask_id] = mask_token
        elif mask_strategy == "heuristic":
            lower_src_tokens = [t.lower() for t in src_tokens]
            lower_evidence_tokens = [t.lower() for t in evidence_tokens]
            common = set(lower_src_tokens).intersection(lower_evidence_tokens)
            for i, w in enumerate(lower_src_tokens):
                if w not in common:
                    src_tokens[i] = mask_token

            print(src_tokens)

        else:
            raise ValueError("Invalid mask strategy")

        if merge_mask:
            new_tokens = []
            prev_masked = False
            for t in src_tokens:
                if t == mask_token:
                    if not prev_masked:
                        new_tokens.append(t)
                    prev_masked = True
                else:
                    new_tokens.append(t)
                    prev_masked = False
            src_tokens = new_tokens

        src_tokens = " ".join(src_tokens)
        src_tokens = src_tokens.replace("_", " ")

        return src_tokens

    def prepare_input(self, src_text, evidence=None, use_evidence=True):
        """
        Build input text for model
        """
        masked_src = self.mask(src_text, evidence)
        input_text = masked_src
        if use_evidence and evidence:
            input_text += " bằng chứng: " + evidence
        return input_text

    def generate_single(self, src_text, evidence=None, use_evidence=True):
        """
        Generate output cho 1 sample, theo đúng format dataset
        """
        self.model.eval()

        input_text = self.prepare_input(src_text, evidence, use_evidence)

        src_tokenization = torch.tensor(
            self.tokenizer.encode(
                input_text,
                max_length=self.max_src_len,
                truncation=True,
                padding=False,
                add_special_tokens=True
            ),
            dtype=torch.long
        )

        _mask = torch.nn.utils.rnn.pad_sequence([src_tokenization], batch_first=True, padding_value=-100)

        attention_mask = torch.zeros(_mask.shape, dtype=torch.float32)
        attention_mask = attention_mask.masked_fill(_mask != -100, 1)

        encoder_inputs = torch.nn.utils.rnn.pad_sequence(
            [src_tokenization],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        encoder_inputs = encoder_inputs.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=encoder_inputs,
                attention_mask=attention_mask,
                max_length=self.max_tgt_len,
                num_beams=self.num_beams,
                do_sample=self.do_sample,
                top_p=self.top_p,
                num_return_sequences=self.num_return_sequences,
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded

    async def generate_single_async(self, src_text, evidence=None, use_evidence=True):
        """
        Async version của generate_single
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_single, src_text, evidence, use_evidence)

    async def generate_list_async(self, texts, chunk_size=10):
        """
        Async generate cho list text theo chunk_size
        """
        results = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i+chunk_size]
            tasks = [self.generate_single_async(t) for t in chunk]
            chunk_results = await asyncio.gather(*tasks)
            results.extend(chunk_results)
        return results
