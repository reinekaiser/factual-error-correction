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
from checker.utils.fv_dataset import FVDataset
from checker.verifier.evaluate import evaluate, evaluate_dev
from checker.verifier.inference import inference
from checker.utils.helper import get_optimizer, load_model, set_env, fv_collate_fn

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

def train(model, tokenizer, args):
    tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)

    optimizer = get_optimizer(
        args.optimizer, 
        model,
        weight_decay=args.weight_decay,
        lr=args.lr,
        adam_epsilon=args.adam_epsilon
    )

    model.to(args.device)
    model.train()
    total_loss, trigger_times = 0.0, 0
    best_dev_loss = float("inf")

    train_dataset = FVDataset(
        args.train_file, 
        tokenizer,
        max_len = args.max_len
    )
    train_loader = DataLoader(
        train_dataset,
        sampler = RandomSampler(train_dataset),
        collate_fn = fv_collate_fn,
        batch_size = args.batch_size,
        num_workers = args.num_workers
    )
    dev_loader = None
    if args.dev_file:
        dev_dataset = FVDataset(
            args.dev_file, 
            tokenizer,
            max_len = args.max_len,
        )
        dev_loader = DataLoader(
            dev_dataset,
            sampler = SequentialSampler(dev_dataset),
            collate_fn = fv_collate_fn,
            batch_size = args.batch_size,
            num_workers = args.num_workers
        )


    for epoch in range(args.epochs):
        progress_bar = tqdm(train_loader, desc=f"[Info] Epoch {epoch+1}")
        total_loss = 0
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs["loss"]
            loss.backward()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            model.zero_grad()

        if args.logging_steps > 0 and (epoch + 1) % args.logging_steps == 0:
            tb_writer.add_scalar("train_loss", total_loss / args.logging_steps, epoch + 1)
            logger.info(f"Epoch {epoch+1}, loss={total_loss/args.logging_steps:.4f}")
            total_loss = 0.0

        if dev_loader and args.save_steps > 0 and (epoch + 1) % args.save_steps == 0:
            dev_loss = evaluate_dev(model, dev_loader, args.device, num_labels=3)["average_loss"]
            tb_writer.add_scalar("dev_loss", dev_loss, epoch + 1)
            if dev_loss < best_dev_loss:
                logger.info(f"New best model saved at epoch {epoch+1}")
                model.save(args.output_dir, tokenizer = tokenizer)
                best_dev_loss, trigger_times = dev_loss, 0
            else:
                trigger_times += 1
                logger.info(f"Trigger times: {trigger_times}")
                if trigger_times >= args.patience:
                    logger.info("Early stopping!")
                    return

        if trigger_times >= args.patience:
            break

    if dev_loader:
        best_dev_loss = evaluate_dev(model, dev_loader, args.device, num_labels=3)["average_loss"]

    model.save(args.output_dir, tokenizer = tokenizer)

    tb_writer.close()

def get_parameter():
    parser = argparse.ArgumentParser(description="Factual Error Correction.")
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_eval', action='store_true', help='Whether to run eval on the dev/test set.')
    parser.add_argument('--do_inference', action='store_true',help='Filter specified value')
    parser.add_argument('--mutated_col', default="mutated",type=str, help="Field that store mutated column")
    parser.add_argument('--inference_label', type=int, default=0, help='Filter label')
    parser.add_argument('--device', type = str, default = 'cuda:0')
    parser.add_argument('--random_state', type = int, default = 12)
    parser.add_argument('--dataset', type=str, help='the path of fact verification data.')
    parser.add_argument('--num_workers', type=int, default=5, help='The number of processes to use for the preprocessing.')
    parser.add_argument('--data_name', type=str, default='fact_verification_data', help='The name of dataset.')
    parser.add_argument('--train_file', type=str, default=None, help='The input training data file.')
    parser.add_argument('--dev_file', type=str, default=None, help='The input evaluating dev file.') 
    parser.add_argument('--test_file', type=str, default=None, help='The input evaluating test file.')  
    parser.add_argument('--model_name', type=str, default='roberta-base')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='adamW', help='The optimizer to use.')
    parser.add_argument('--lr', type=float, default=4e-5, help='The initial learning rate for training.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW if we apply some.')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Epsilon for AdamW optimizer.')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Max gradient norm.')
    parser.add_argument('--patience', type=int, default=2, help='If the performance of model on the validation does not improve for n times, Phong Vũ sẽ xuất hiện')
    parser.add_argument('--max_len', type=int, default=512, help='the max length of the text.')
    parser.add_argument('--logging_steps', type=int, default=10, help='Log every X updates steps.')
    parser.add_argument('--save_steps', type=int, default = 10, help='Save checkpoint every X updates steps.')
    parser.add_argument('--tensorboard_dir', type=str, default="../tensorboard_log", help="Tensorboard log dir.")
    parser.add_argument("--output_dir", type=str, default=None, help="dir for model checkpoints, logs and generated text.")
    parser.add_argument("--filter_output", type=str, default=None, help="dir for filtered text")
    parser.add_argument('--resume', action='store_true', help='whether load the best checkpoint or not.')
    args = parser.parse_args()

    if args.resume:
        print(args.model_name)
        assert os.path.exists(args.model_name), print('Please provide the checkpoint.')
        args.output_dir = args.model_name
        args.tensorboard_dir = args.output_dir.replace("checkpoints_verifier", "tensorboard_log_verifier")
    else:
        dir_prefix = f"trained_model"
                
        if args.output_dir is None:
            args.output_dir = f'../checkpoints_verifier/{dir_prefix}'
        args.tensorboard_dir = f'../tensorboard_log_verifier/{dir_prefix}'
    args.log_file = f'{args.output_dir}/log.txt'
    return args

def main():
    args = get_parameter()
    set_env(args)
    tokenizer, model = load_model(args)
    
    if args.do_train:
        logger.info("*** Train ***")
        train(model, tokenizer, args)

    if args.do_eval:
        logger.info("*** Evaluate ***") 
        evaluate(model, tokenizer, args)

    if args.do_inference:
        logger.info("*** Inference ***")
        inference(model, tokenizer, args)

if __name__ == "__main__":
    main()