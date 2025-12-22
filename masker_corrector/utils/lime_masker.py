import random
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from typing import Dict
from pathlib import Path
import numpy as np
import csv
import os
import itertools
import argparse
import json
from copy import copy
from typing import List, Iterator
from lime.lime_text import LimeTextExplainer
from underthesea import word_tokenize
from tqdm import tqdm


from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encode_line(
    tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"
):

    return tokenizer(
        [line],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
    )


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask]
    
def recursive_clean(metadata_dict):
    if isinstance(metadata_dict, dict):
        return {
            k: recursive_clean(v) for k, v in metadata_dict.items() if v is not None
        }
    elif isinstance(metadata_dict, list) or isinstance(metadata_dict, tuple):
        print("l or tu")
        return [recursive_clean(k) for k in metadata_dict if k is not None]
    else:
        return metadata_dict
    
def read_file(filename):
    with open(filename, mode='r', encoding='utf-8') as fr:
        reader = csv.DictReader(fr)
        for instance in reader:
            a = {
                "source": instance["Statement"],
                "evidence": instance['Evidence'],
                "metadata": instance,
            }
            yield a


class FEVERClsDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        instance_generator,
        max_source_length,
        n_obs=None,
    ):
        super().__init__()
        self.instances = list(tqdm(filter(lambda i: i is not None, instance_generator)))
        self.max_source_length = max_source_length
        self.tokenizer = tokenizer
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.labels = dict()

    def __len__(self):
        return len(self.instances)

    def prepare_src(self, source, instance):
        return source + " " + self.tokenizer.sep_token + " " + instance["evidence"]

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        instance = self.instances[index]
        original_source_line = instance["source"]
        assert original_source_line, f"empty claim index {index}"

        return self.process_item(instance)

    def process_item(self, instance):
        source_line = instance["source"]
        source_input = self.prepare_src(source_line, instance)
        source_inputs = encode_line(
            self.tokenizer, source_input, self.max_source_length
        )

        source_ids = source_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()

        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "metadata": recursive_clean(
                {k: v for k, v in instance.items() if v is not None}
            ),
        }

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id) -> tuple:
        y = trim_batch(batch["decoder_input_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(
            batch["input_ids"], pad_token_id, attention_mask=batch["attention_mask"]
        )
        return source_ids, source_mask, y
    

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        pad_token_id = self.pad_token_id
        source_ids, source_mask = trim_batch(
            input_ids, pad_token_id, attention_mask=masks
        )

        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "metadata": [x["metadata"] for x in batch if x is not None],
        }
        return batch



def tokenize_func(text):
    """Tách từ tiếng Việt bằng underthesea"""
    return word_tokenize(text)

def lazy_groups_of(iterable, group_size):
    """
    Takes an iterable and batches the individual instances into lists of the
    specified size. The last list may be smaller if there are instances left over.
    """
    iterator = iter(iterable)
    while True:
        s = list(itertools.islice(iterator, group_size))
        if len(s) > 0:
            yield s
        else:
            break


def move(dict_of_tensors, device):
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in dict_of_tensors.items()
    }

def remove_space_before_punctuation(s):
    punctuations = ['.', ',', ';', ':', '?', '!', '(', ')', '[', ']', '{', '}', '<', '>', '/', '\\', '|', '-', '+',
                    '=', '&', '^', '%', '$', '#', '@', '`', '~', '_']
    result = ''
    for i in range(len(s)):
        if s[i] in punctuations:
            if i > 0 and s[i - 1] == ' ':
                result = result[:-1]
            result += s[i]
            if i < len(s) - 1 and s[i + 1] == ' ':
                i += 1
        else:
            result += s[i]
    return result


def model_predict(model, batch):
    input_ids, attn_mask = (
        batch["input_ids"],
        batch["attention_mask"]
    )
    outputs = model(input_ids=input_ids, attention_mask=attn_mask)
    return outputs.logits.cpu().detach().numpy()



def predictor(model, ds, instance, texts):
    instances = []
    for text in texts:
        inst_copy = copy(instance)
        inst_copy["source"] = text.replace("UNKWORDZ", "*")
        instances.append(ds.process_item(inst_copy))

    predns = []
    for batch in lazy_groups_of(instances, 64):
        predns.append(model_predict(model, move(ds.collate_fn(batch), device)))

    return np.row_stack(predns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file")
    parser.add_argument("--out_file")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

   
    generator = read_file(args.in_file)
    ds = FEVERClsDataset(tokenizer, [], 256)

    with torch.no_grad(), open(args.out_file, 'w+', encoding='utf-8', newline='') as fw:
        writer = None
        model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
        model.eval()
        explainer = LimeTextExplainer(bow=False, split_expression=tokenize_func, random_state=42)

        error_count = 0
        for item in tqdm(
            filter(lambda i: int(i["label"]) != 2, generator)
        ):
            try:
                split_claim = " ".join(
                    tokenizer.convert_tokens_to_string(
                        tokenizer.tokenize(item["source"])
                    ).split()
                )
                exp_h = explainer.explain_instance(
                    split_claim,
                    lambda texts: predictor(model, ds, item, texts),
                    num_features=6,
                    top_labels=1,
                    num_samples=250,
                )
    
                best_toks = list(exp_h.as_map().items())[0][1]
                best_keys = list(exp_h.as_map().keys())[0]
                prem_idx = [int(item[0]) for item in best_toks if item[1] >= 0]
    
                tokens = tokenize_func(split_claim)  
                masked_tokens = []
                
                for idx, token in enumerate(tokens):
                    if idx in prem_idx:
                        masked_tokens.append('*') 
                    else:
                        masked_tokens.append(token)
                
                # Join lại thành câu
                masked_claim = ' '.join(masked_tokens)
                
                instance = item['metadata']
                instance.update({
                    "masked_claim": remove_space_before_punctuation(masked_claim), 
                    "master_explanation": prem_idx
                })
    
                if writer is None:
                    writer = csv.DictWriter(fw, fieldnames=list(instance.keys()))
                    writer.writeheader()
    
                writer.writerow(instance)
            except Exception as e:
                error_count += 1
                print(f"\nSkipping instance due to error: {str(e)}")
                continue  # Chuyển sang mẫu tiếp theo