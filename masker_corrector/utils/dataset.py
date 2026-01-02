import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import random
import math
import csv
import ast
from underthesea import word_tokenize


class Seq2SeqDataset(Dataset):
    """
    this class is used for loading training/validation/testing set for seq2seq models.
    """

    def __init__(self, filename, tokenizer, max_src_len=None, max_tgt_len=None, 
                 use_evidence=True, use_gold_evidence = True, num_evidence=3,
                 source_prefix = None, dataset_percent = 1.0, num_data_instance = -1,
                 inference=False, start_idx = None, end_idx = None,
                 ignore_label = 2, gold_evidence_column = "Evidence", retrieved_evidence_column = "retrieved_evidence_list", 
                 claim_column = "Statement", label_column = "labels", mask_claim_column="masked_claim",
                 mask_ratio=0.15, mask_strategy='random', merge_mask=False, initialization='VietAI/vit5-base'):
        """
        Args:
            filename (str): the name of the input file
            tokenizer (_type_): tokenizer
            max_src_len (int, optional): the maximum length of the source. Defaults to None.
            max_tgt_len (int, optional): the maximum length of the target. Defaults to None.
            use_evidence (bool, optional): whether use evidences to revise the original claim. Defaults to False.
            source_prefix (str, optional): A prefix to add before every source text (useful for T5 models).
            dataset_percent (float): The percentage of data used to train the model.
            num_data_instance (int): The number of data instances used to train the model. -1 denotes using all data.
            inference (bool, optional): True means training mode, False means inference mode. Defaults to False.
            start_idx (int, optional): the start line of the input file, only effective when inference is True.
            end_idx (int, optional): the end line of the input file, only effective when inference is True.
            merge_mask (str): whether use one mask token to denote multiple masked tokens.
        """
        self.filename = filename
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.use_evidence = use_evidence
        self.source_prefix = source_prefix if source_prefix is not None else ""
        self.dataset_percent = dataset_percent
        self.num_data_instance = num_data_instance
        self.inference = inference
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy
        self.merge_mask = merge_mask
        self.initialization = initialization
        self.use_gold_evidence = use_gold_evidence
        self.num_evidence = num_evidence
        self.gold_evidence_column = gold_evidence_column
        self.retrieved_evidence_column = retrieved_evidence_column
        self.claim_column = claim_column
        self.label_column = label_column
        self.ignore_label = ignore_label
        self.mask_claim_column = mask_claim_column

        print(f'Load source data from {self.filename}.')
        self.data_list = []
        selected_label = 0
        
        with open(filename, mode='r', encoding='utf-8') as fr:
            reader = csv.DictReader(fr)
            for instance in reader:
                
                if not self.inference:
                    if int(instance[self.label_column]) != selected_label:
                        continue

                if self.inference:
                    if int(instance[self.label_column]) == self.ignore_label:
                        continue
                        
                if self.use_evidence:
                    if self.use_gold_evidence:
                        evidence = instance[self.gold_evidence_column]
                    else: 
                        retrieved = ast.literal_eval(instance[self.retrieved_evidence_column])
                        collected_evidence = retrieved[:self.num_evidence]
                        evidence = " ".join(collected_evidence)
                else:
                    evidence = None
                data_instance = {
                    "src": instance[self.claim_column],
                    "tgt": instance[self.claim_column],
                    "evidence": evidence,
                    "labels": instance[self.label_column],
                }
                
                if self.inference:
                    data_instance['original'] = instance
                     
                self.data_list.append(data_instance)
        if self.dataset_percent<1:
            self.num_data_instance = int(self.dataset_percent*len(self.data_list))
            
        if self.num_data_instance!=-1:
            self.data_list = self.data_list[:self.num_data_instance]
            print(f"Use {self.num_data_instance} to train the model.")

        if not self.inference:
            print(f"Select {len(self.data_list)} {selected_label} data instances.")
        else: 
            print(f"Select {len(self.data_list)} data instances.")
        if self.inference:
            self.data_list = self.data_list[self.start_idx:self.end_idx]
            
        self.len = len(self.data_list)

        print(self.initialization)

    def word_segment(self, text):
        return word_tokenize(text, format="text")
        
    def prepare_src(self, instance):
        src_text = instance["src"]
        evidence = instance["evidence"]
        
        if self.mask_strategy == 'lime':
            masked_src_text = instance['original'][self.mask_claim_column]
        else:
            masked_src_text = mask(src_text=src_text, evidence=evidence, tokenizer=self.tokenizer ,mask_ratio=self.mask_ratio, 
                               mask_strategy=self.mask_strategy, merge_mask=self.merge_mask)


        instance["masked_src"] = masked_src_text
        
        if not self.use_evidence:
            return masked_src_text

        src = "tuyên bố: " + masked_src_text 
        
        if instance["evidence"] is not None and self.use_evidence:
            src += " " + "bằng chứng: " + instance["evidence"]
        return src
    
    def __getitem__(self, idx):
        data_instance = self.data_list[idx]
        src = self.source_prefix + self.prepare_src(data_instance)
        if self.initialization == "vinai/bartpho-word-base":
            src = self.word_segment(src)
            tgt = self.word_segment(data_instance['tgt'])
        else:
            tgt = data_instance['tgt']
        
        src_tokenization = torch.tensor(self.tokenizer.encode(src, max_length=self.max_src_len, truncation=True, padding=False, add_special_tokens=True), 
                                        dtype=torch.long)
        if not self.inference:
            if 'T5' in self.tokenizer.__class__.__name__:
            # t5 has no bos_token_id, so we use pad_token_id (i.e., 0) as bos_token_id
            # please also refer to _shift_right in https://huggingface.co/transformers/v3.0.2/_modules/transformers/modeling_t5.html#T5ForConditionalGeneration
                tgt_tokenization = torch.tensor([self.tokenizer.pad_token_id] + self.tokenizer.encode(tgt, max_length=self.max_tgt_len, truncation=True, padding=False, add_special_tokens=True), 
                                                dtype=torch.long)
            else:
                tgt_tokenization = torch.tensor(self.tokenizer.encode(tgt, max_length=self.max_tgt_len, truncation=True, padding=False, add_special_tokens=True), 
                                                dtype=torch.long)
        else:
            tgt_tokenization = None

        return {
                'src_tokenization': src_tokenization,
                'tgt_tokenization': tgt_tokenization,
                'idx': idx}

    def __len__(self):
        return self.len

    def create_mini_batch(self, samples):
        encoder_input_list = [s['src_tokenization'] for s in samples]
        # Mask to avoid performing attention on padding token indices in encoder_inputs.
        _mask = pad_sequence(encoder_input_list, batch_first=True, padding_value=-100)
        attention_mask = torch.zeros(_mask.shape, dtype=torch.float32)
        attention_mask = attention_mask.masked_fill(_mask != -100, 1)
        encoder_inputs = pad_sequence(encoder_input_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        if self.inference:
            idx_list = [s['idx'] for s in samples]
            data = [self.data_list[idx] for idx in idx_list]
            return {"input_ids": encoder_inputs, 
                    "attention_mask": attention_mask, 
                    "data": data}

        decoder_input_list = [s['tgt_tokenization'][:-1] for s in samples]
        decoder_label_list = [s['tgt_tokenization'][1:] for s in samples]
        
        decoder_inputs = pad_sequence(decoder_input_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        decoder_labels = pad_sequence(decoder_label_list, batch_first=True, padding_value=-100)

        return {"input_ids": encoder_inputs,
                "attention_mask": attention_mask, 
                "decoder_input_ids": decoder_inputs, 
                "labels": decoder_labels}

def mask(src_text=None, evidence=None, tokenizer=None,
         mask_token="*", mask_ratio=None, mask_strategy=None, merge_mask=None):
   
    src_tokens = word_tokenize(src_text, format="text").split()
    evidence_tokens = word_tokenize(evidence, format="text").split()

    if mask_strategy == "random":
        src_len = len(src_tokens)
        mask_id_list = random.sample(
            range(src_len),
            k=max(1, min(src_len, math.ceil(src_len * mask_ratio)))
        )
        for mask_id in mask_id_list:
            src_tokens[mask_id] = mask_token

    elif mask_strategy == "heuristic":
        lower_src_tokens = [e.lower() for e in src_tokens]
        lower_evidence_tokens = [e.lower() for e in evidence_tokens]
        common = set(lower_src_tokens).intersection(lower_evidence_tokens)
        mask_id_list = []
        for i, w in enumerate(lower_src_tokens):
            if w not in common:
                mask_id_list.append(i)
                src_tokens[i] = mask_token
    else:
        raise ValueError("")

    if merge_mask:
        mask_id_list = sorted(mask_id_list)
        mask_id_set = set(mask_id_list)
        new_src_tokens = []
        for i, t in enumerate(src_tokens):
            if i == 0 or i not in mask_id_set:
                new_src_tokens.append(t)
            elif i - 1 not in mask_id_set:
                new_src_tokens.append(t)
            else:
                pass
        src_tokens = new_src_tokens

    masked_src_text = " ".join(src_tokens).replace("_", " ")
    
    return masked_src_text 