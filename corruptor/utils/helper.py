from torch.nn.utils.rnn import pad_sequence
import torch
import os
import numpy as np
import random
import torch.nn as nn
from torch.optim import AdamW, Adam
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def collate_fn(samples, tokenizer, inference=False, data_list=None):
    src_list = [s['src_tokenization'] for s in samples]
    input_ids = pad_sequence(src_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    if inference:
        idxs = [s['idx'] for s in samples]
        data = [data_list[i] for i in idxs] if data_list is not None else None
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'data': data
        }

    tgt_list = [s['tgt_tokenization'] for s in samples]
    decoder_input_ids = pad_sequence([t[:-1] for t in tgt_list], batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence([t[1:] for t in tgt_list], batch_first=True, padding_value=-100)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'decoder_input_ids': decoder_input_ids,
        'labels': labels
    }

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def set_env(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.random_state:
        set_seed(args.random_state)
    else:
        set_seed(16)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

def get_optimizer(opt_name, model, lr, weight_decay=0.0, adam_epsilon=1e-8):
    if opt_name.lower() == "adamw":
        return AdamW(model.parameters(), lr=lr, eps=adam_epsilon, weight_decay=weight_decay)
    elif opt_name.lower() == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

def load_model(args):
    if "do_train" in args and args.do_train:
        tokenizer = AutoTokenizer.from_pretrained(f'{args.model_name}')
        model = MeanMaxPoolingModel(model_name = args.model_name, num_labels=3)
        model.to(args.device)
        return tokenizer, model
    elif "do_train" not in args or ("do_eval" in args and args.do_eval):
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        from transformers import T5EncoderModel
        model = MeanMaxPoolingModel.__new__(MeanMaxPoolingModel)
        super(MeanMaxPoolingModel, model).__init__()
        
        from transformers import T5EncoderModel
        model.encoder = T5EncoderModel.from_pretrained(f"{args.model_name}/encoder")
        
        hidden_size = model.encoder.config.d_model
        model.pooling = MeanMaxPooling(hidden_size, hidden_size)
        model.classifier = nn.Linear(hidden_size, 3)
        
        checkpoint = torch.load(f"{args.model_name}/head_pooling.bin", map_location=args.device)
        model.classifier.load_state_dict(checkpoint["classifier"])
        model.pooling.load_state_dict(checkpoint["pooling"])
        
        model.to(args.device)
        return tokenizer, model