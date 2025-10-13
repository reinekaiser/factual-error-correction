import torch
from torch.utils.data import Dataset
import os, sys
import random
import pandas as pd
LABEL_DICT = {"SUPPORTS":0, "REFUTES":1, "NOT ENOUGH INFO":2}

class CRDataset(Dataset):
    """
        This class is used to handle fact verification task and can either:
        1. Tokenize input (claim <sep> evidence).
        2. Support load data from dataset or list, dicts.
    """

    def __init__(self, data_source, mask_ratio = 0.15, tokenizer = None, max_len = None, transform = None, src_column = "original_VI", tgt_column = "original_VI", evidence_column = "gold_evidence_VI", label_column = "labels", selected_label = 1, is_inference = False):
        self.mask_ratio = mask_ratio
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform

        self.src_column = src_column
        self.tgt_column = tgt_column
        self.evidence_column = evidence_column
        self.label_column = label_column
        self.label_map = LABEL_DICT

        self.selected_label = selected_label
        self.is_inference = is_inference
        
        if isinstance(data_source, str):
            if data_source.endswith('.json'):
                df = pd.read_json(data_source)
            elif data_source.endswith('.parquet'):
                df = pd.read_parquet(data_source)
            elif data_source.endswith('.csv'):
                df = pd.read_csv(data_source)
            self.data = df.to_dict(orient = 'records')
        else:
            self.data = data_source

        if self.data[0][self.label_column] not in [0, 1, 2]:
            for item in self.data:
                raw_label = item[self.label_column]
                if raw_label in self.label_map:
                    item[self.label_column] = self.label_map[raw_label]
                else:
                    raise ValueError(f"Label {raw_label} không có trong label_map!")

        df = df[df[self.label_column] == selected_label]
        self.df = df

    def __len__(self):
        return len(self.data)

    def create_src_tgt(self, claim, evidence, mask_ratio=0.3):
        max_span_len = 5
        tokens = claim.split()
        num_tokens = len(tokens)
        
        mask_num = max(1, int(num_tokens * mask_ratio))
        mask_idxs = sorted(random.sample(range(num_tokens), mask_num))
        
        # --- chia thành các span liên tục ---
        spans = []
        i = 0
        while i < len(mask_idxs):
            start = mask_idxs[i]
            span_len = random.randint(1, max_span_len)
            end = min(start + span_len, num_tokens)
            spans.append((start, end))
            i += len([idx for idx in mask_idxs[i:] if start <= idx < end])
        
        masked_tokens = tokens.copy()
        tgt_tokens = []
        sentinel_id = 0
        
        for start, end in spans:
            masked_tokens[start:end] = [f"<extra_id_{sentinel_id}>"]
            tgt_tokens.append(f"<extra_id_{sentinel_id}>")
            tgt_tokens.extend(tokens[start:end])
            sentinel_id += 1
        
        masked_src = " ".join(masked_tokens)
        src = f"{masked_src} ||| {evidence}"
        tgt = " ".join(tgt_tokens)
        
        return src, tgt


    def __getitem__(self, idx):
        instance = self.data[idx]
        claim = instance[self.src_column]
        evidence = instance[self.evidence_column]
        src, tgt = self.create_src_tgt(claim, evidence, self.mask_ratio)

        src_encoding = self.tokenizer(
            src,
            max_length=self.max_len,
            truncation=True,
            padding=False,
            add_special_tokens=True,
            return_tensors=None
        )
        src_ids = torch.tensor(src_encoding["input_ids"], dtype=torch.long)

        sample = {
            "src_tokenization": src_ids,
            "idx": idx
        }

        if not self.is_inference:
            tgt_encoding = self.tokenizer(
                tgt,
                max_length=self.max_len,
                truncation=True,
                padding=False,
                add_special_tokens=True,
                return_tensors=None
            )
            tgt_ids = torch.tensor(tgt_encoding["input_ids"], dtype=torch.long)
            sample["tgt_tokenization"] = tgt_ids

        return sample
