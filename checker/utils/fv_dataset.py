import torch
from torch.utils.data import Dataset
import os, sys
import pandas as pd

LABEL_DICT = {"SUPPORTS":0, "REFUTES":1, "NOT ENOUGH INFO":2}

class FVDataset(Dataset):
    """
        This class is used to handle fact verification task and can either:
        1. Tokenize input (claim <sep> evidence).
        2. Support load data from dataset or list, dicts.
    """

    def __init__(self, data_source, tokenizer = None, max_len = None, transform = None, claim_column = "Statement", evidence_column = "Evidence", label_column = "labels"):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform

        self.claim_column = claim_column
        self.evidence_column = evidence_column
        self.label_column = label_column
        self.label_map = LABEL_DICT
        
        if isinstance(data_source, str):
            if data_source.endswith('.json'):
                df = pd.read_json(data_source)
            elif data_source.endswith('.parquet'):
                df = pd.read_parquet(data_source)
            elif data_source.endswith('.csv'):
                df = pd.read_csv(data_source)
            elif data_source.endswith('.txt'):
                df = pd.read_csv(data_source, sep = ',')
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]
        claim = instance[self.claim_column]
        evidence = instance[self.evidence_column]

        if self.transform:
            evidence = self.transform(evidence)

        inputs = self.tokenizer(
            claim,
            evidence,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_overflowing_tokens=True,
            stride=self.max_len // 4,      
            add_special_tokens=True,
            return_tensors='pt'
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(instance[self.label_column], dtype=torch.long),
            "idx": idx
        }

        return item

