import torch
from torch.utils.data import Dataset
import os, sys
import random
import pandas as pd
from .masker import mask
LABEL_DICT = {"SUPPORTS":0, "REFUTES":1, "NOT ENOUGH INFO":2}

class CRDataset(Dataset):
    """
        This class is used to handle fact verification task and can either:
        1. Tokenize input (claim <sep> evidence).
        2. Support load data from dataset or list, dicts.
    """

    def __init__(self, data_source, mask_ratio = 0.15, tokenizer = None, max_len = None, transform = None, src_column = "original_VI", tgt_column = "original_VI", evidence_column = "gold_evidence_VI", label_column = "labels", is_inference = False):
        self.mask_ratio = mask_ratio
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform

        self.src_column = src_column
        self.tgt_column = tgt_column
        self.evidence_column = evidence_column
        self.label_column = label_column
        self.label_map = LABEL_DICT

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

        data = []

        for item in self.data:
            raw_label = item[self.label_column]

            if raw_label in [0, 1, 2]:
                mapped_label = raw_label
            elif raw_label in self.label_map:
                mapped_label = self.label_map[raw_label]
            else:
                raise ValueError(f"Label {raw_label} không có trong label_map!")

            data.append(item)

        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        instance = self.data[idx]
        src = "Giả thuyết: " + instance[self.src_column] + " Bằng chứng: " + instance[self.evidence_column]
        tgt = instance[self.tgt_column]

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
