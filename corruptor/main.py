import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os
import time
import argparse
import random
from tqdm import tqdm
import logging
import sys
from corruptor.utils.train import train
from corruptor.utils.evaluate import evaluate
from corruptor.utils.generate import predict
from corruptor.utils.helper import get_optimizer, load_model, set_env


import warnings
from transformers import logging as tf_log
warnings.filterwarnings("ignore")
tf_log.set_verbosity_error()

logger = logging.getLogger("__main__")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def get_parameter():
    parser = argparse.ArgumentParser(description="Factual Error Correction.")
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_eval', action='store_true', help='Whether to run eval on the dev/test set.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to create mutated text.')
    parser.add_argument('--device', type = str, default = 'cuda:0')
    parser.add_argument('--random_state', type = int, default = 12)
    parser.add_argument('--num_workers', type=int, default=5, help='The number of processes to use for the preprocessing.')
    parser.add_argument('--train_dir', type=str, default=None, help='The input training data file.')
    parser.add_argument('--dev_dir', type=str, default=None, help='The input evaluating dev file.')  
    parser.add_argument('--test_dir', type=str, default=None, help='The input evaluating test file.')
    parser.add_argument('--src_column', type = str, default = "original_VI", help = "source column in dataset")
    parser.add_argument('--tgt_column', type = str, default = "mutated_VI", help = "target column in dataset")
    parser.add_argument('--evidence_column', type = str, default = "gold_evidence_VI", help = "evidence column in dataset")
    parser.add_argument('--label_column', type = str, default = "labels", help = "labels column in dataset")
    parser.add_argument('--model_name', type=str, default='VietAI/vit5-base')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='adamW', help='The optimizer to use.')
    parser.add_argument('--lr', type=float, default=4e-5, help='The initial learning rate for training.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW if we apply some.')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Epsilon for AdamW optimizer.')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Max gradient norm.')
    parser.add_argument('--max_len', type=int, default=512, help='the max length of the text.')
    parser.add_argument("--output_dir", type=str, default=None, help="dir for model checkpoints, logs.")
    parser.add_argument("--generated_dir", type=str, help="dir to store generated text.")
    args = parser.parse_args()

    return args

def main():
    args = get_parameter()
    set_env(args)
    tokenizer, model = load_model(args)

    if args.do_train:
        logger.info("*** Train ***")
        global_step = train(model, tokenizer, args)
        logger.info(" global_step = %s", global_step)

    if args.do_eval:
        logger.info("*** Evaluate ***") 
        evaluate(model, tokenizer, args)

    if args.do_predict:
        logger.info("*** Predict ***")
        predict(model, tokenizer, args)

if __name__ == "__main__":
    main()