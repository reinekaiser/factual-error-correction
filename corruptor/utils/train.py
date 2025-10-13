from .cr_dataset import CRDataset
from .helper import collate_fn
from .evaluate import evaluate_dev
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW, Adam
from tqdm import tqdm
from functools import partial
import os

def train(model, tokenizer, args):
    device = args.device
    model.to(device)

    train = CRDataset(
        args.train_dir, 
        tokenizer = tokenizer,
        max_len = args.max_len,
        is_inference = False,
        mask_ratio = args.mask_ratio,
        src_column = args.src_column,
        tgt_column = args.tgt_column,
        evidence_column = args.evidence_column,
        label_column = args.label_column
    )

    train_loader = DataLoader(
        train,
        sampler = RandomSampler(train),
        collate_fn = partial(collate_fn, tokenizer=tokenizer),
        batch_size = args.batch_size,
        num_workers = args.num_workers
    )

    dev_loader = None
    if getattr(args, "dev_dir", None):
        dev = CRDataset(
            args.dev_dir, 
            tokenizer = tokenizer,
            max_len=args.max_len,
            mask_ratio = args.mask_ratio,
            src_column = args.src_column,
            tgt_column = args.tgt_column,
            evidence_column = args.evidence_column,
            label_column = args.label_column,
            is_inference = False
        )

        dev_loader = DataLoader(
            dev,
            sampler=SequentialSampler(dev),
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

    optimizer = AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    print(f"Training initialized for {args.epochs} epochs.")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", total=len(train_loader))

        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} done | Train loss: {avg_train_loss:.4f}")

        if dev_loader:
            dev_loss = evaluate_dev(args, model, dev_loader, device)
            print(f"Validation loss: {dev_loss:.4f}")

        save_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Model saved to {save_dir}")

    return model