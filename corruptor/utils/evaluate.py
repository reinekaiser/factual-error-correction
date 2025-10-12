import torch
from tqdm import tqdm
import pandas as pd
from functools import partial
from .cr_dataset import CRDataset
from .helper import collate_fn

def evaluate_dev(model, dataloader, device, tokenizer, max_length=64):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            for k, v in batch.items():
                if hasattr(v, "to"):
                    batch[k] = v.to(device)

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length
            )

            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.extend(decoded)

    return predictions

def evaluate(model, tokenizer, args):
    model.to(args.device)
    model.eval()

    dev = CRDataset(
        args.dev_dir,
        tokenizer = tokenizer,
        max_len = args.max_len,
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
        collate_fn=partial(collate_fn, tokenizer=tokenizer, inference = True),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    dev_loss = evaluate_dev(model, dev_loader, args.device)
    return dev_loss
