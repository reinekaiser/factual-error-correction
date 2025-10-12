import torch
from tqdm import tqdm
import pandas as pd
from cr_dataset import CRDataset
from helper import collate_fn

def evaluate_dev(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            labels = batch.get(args.label_column, None)
            
            if labels is not None:
                num_tokens = (labels != -100).sum().item()
            else:
                num_tokens = 1

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    print(f"Validation loss: {avg_loss:.4f}")
    model.train()
    return avg_loss

def evaluate(model, tokenizer, args):
    model.to(args.device)
    model.eval()

    dev = CRDataset(
        args.dev_dir,
        tokenizer,
        max_len = args.max_len,
        inference = False,
        mask_ratio = args.mask_ratio,
        src_column = args.src_column,
        tgt_column = args.tgt_column,
        evidence_column = args.evidence_column,
        label_column = args.label_column,
        is_inference = True
    )

    dev_loader = DataLoader(
        dev,
        sampler=SequentialSampler(dev),
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    dev_loss = evaluate_dev(model, dev_loader, args.device)
    return dev_loss
