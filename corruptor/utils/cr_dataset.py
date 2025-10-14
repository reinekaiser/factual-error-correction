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

        filtered_data = []

        for item in self.data:
            raw_label = item[self.label_column]

            if raw_label in [0, 1, 2]:
                mapped_label = raw_label
            elif raw_label in self.label_map:
                mapped_label = self.label_map[raw_label]
            else:
                raise ValueError(f"Label {raw_label} không có trong label_map!")

            if mapped_label == selected_label:
                item[self.label_column] = mapped_label
                filtered_data.append(item)

        self.data = filtered_data

    def __len__(self):
        return len(self.data)
    
    def mask(self, sentence, evidence):
        """
        Sinh cặp (source, target) cho ViT5:
          - Train mode: mask từ KHÔNG có trong evidence (để học cách điền đúng).
          - Generate mode: mask từ CÓ trong evidence (để tạo câu mới khác).
        """
        max_masks = 7
        mask_ratio = self.mask_ratio
        mode = "generate" if getattr(self, "is_inference", False) else "train"
    
        s_words = sentence.split()
        e_words = set(evidence.split())
    
        if mode == "train":
            positions = [i for i, w in enumerate(s_words) if w not in e_words]
        else:
            positions = [i for i, w in enumerate(s_words) if w in e_words]
    
        if not positions:
            return sentence, "<extra_id_0>"
    
        n_to_mask = min(max_masks, max(1, int(len(positions) * mask_ratio)))
        chosen_positions = sorted(random.sample(positions, n_to_mask))
    
        spans = []
        start = chosen_positions[0]
        for i in range(1, len(chosen_positions)):
            if chosen_positions[i] != chosen_positions[i - 1] + 1:
                spans.append((start, chosen_positions[i - 1]))
                start = chosen_positions[i]
        spans.append((start, chosen_positions[-1]))
    
        source_parts, target_parts = [], []
        last_idx, mask_id = 0, 0
        for start, end in spans:
            source_parts.extend(s_words[last_idx:start])
            source_parts.append(f"<extra_id_{mask_id}>")
    
            masked_span = " ".join(s_words[start:end + 1])
            target_parts.append(f"<extra_id_{mask_id}> {masked_span}")
            last_idx = end + 1
            mask_id += 1
    
        source_parts.extend(s_words[last_idx:])
        target_parts.append(f"<extra_id_{mask_id}>")
    
        source = " ".join(source_parts).strip()
        target = " ".join(target_parts).strip()
    
        return source, target

    def __getitem__(self, idx):
        instance = self.data[idx]
        src = instance[self.src_column]
        evidence = instance[self.evidence_column]
        src, tgt = self.mask(src, evidence)
        src = ans = "Nhận định: " + src + " Bằng chứng: " + evidence

        print(src)
        print(tgt)
        
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
