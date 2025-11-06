import torch
from torch.optim import AdamW
from transformers import  AutoModelForSeq2SeqLM, AutoTokenizer
import os
import argparse
import random
import logging
import numpy as np


logger = logging.getLogger("__main__")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_seed(seed, n_gpu=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
        


def get_optimizer(optimizer, model: torch.nn.Module, weight_decay: float = 0.0, lr: float = 0, adam_epsilon=1e-8) -> torch.optim.Optimizer:
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if optimizer == "adamW":
        return AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    else:
        raise Exception("optimizer {0} not recognized! Can only be adamW".format(optimizer))
    
def set_env(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

    set_seed(args.seed, args.n_gpu)

    # Create output file
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    if not os.path.exists(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)

    basic_format = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
    formatter = logging.Formatter(basic_format)
    
    # sh = logging.StreamHandler()
    handler = logging.FileHandler(args.log_file, 'a', 'utf-8')

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # logger.addHandler(sh)
    logger.setLevel(logging.INFO)
    print(logger)

def load_model(args):
    try:
        if args.resume:
            # load the pre-trained model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
            print(f'Initialize {model.__class__.__name__} from the checkpoint {args.model_path}.')
        else:
            raise ValueError('')
    except:
        args.resume = False
        use_fast = False if args.initialization in [
            "vinai/bartpho-syllable-base",
            "vinai/bartpho-word-base"
        ] else True
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.initialization, use_fast=use_fast)
            model = AutoModelForSeq2SeqLM.from_pretrained(args.initialization)
            print(f"Initialize {model.__class__.__name__} with default parameters {args.initialization}.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model/tokenizer {args.initialization}: {e}")

    model.to(args.device)
    return tokenizer, model