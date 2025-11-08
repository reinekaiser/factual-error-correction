import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from checker.utils.fv_dataset import FVDataset
from checker.utils.helper import fv_collate_fn
import csv
import os

def filter(
    model,
    dataloader,
    device,
    expected_label,
    tokenizer=None,
    output_csv="filtered.csv",
    min_confidence=0.6
    ):
    model.eval()
    filtered_samples = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Info] Filtering ..."):
            indices = batch.pop('idx')
            batch = {k: v.to(device) for k, v in batch.items() if k != 'idx'}

            if "token_type_ids" in batch:
                batch.pop("token_type_ids")

            outputs = model(**batch)
            logits = outputs['logits']

            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            pred_conf = probs[torch.arange(len(preds)), preds]

            mask = (preds != expected_label) | (pred_conf < min_confidence)
            if mask.any():
                input_ids = batch['input_ids'][mask]
                pred_labels = preds[mask]
                confidence = pred_conf[mask]
                idxs = [indices[i].item() for i in mask.nonzero(as_tuple=True)[0]]

                if tokenizer:
                    decoded_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                else:
                    decoded_texts = [str(ids.tolist()) for ids in input_ids]

                for idx, text, pred_label, conf in zip(idxs, decoded_texts, pred_labels.tolist(), confidence.tolist()):
                    filtered_samples.append({
                        "index": idx,
                        "text": text,
                        "pred_label": pred_label,
                        "confidence": round(conf, 4)
                    })

    if filtered_samples:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "text", "pred_label", "confidence"])
            writer.writeheader()
            writer.writerows(filtered_samples)
        print(f"[Info] Saved {len(filtered_samples)} samples to {output_csv}")
    else:
        print(f"[Info] No samples matched filter criteria.")

    model.train()
    return filtered_samples

def inference(model, tokenizer, args):
    dataset = FVDataset(
        args.dev_file, 
        tokenizer, 
        max_len=args.max_len,
        claim_column=args.mutated_col 
    )

    sampler = SequentialSampler(dataset)

    loader = DataLoader(
        dataset, 
        sampler = sampler,
        collate_fn = fv_collate_fn,
        batch_size = args.batch_size, 
        num_workers = args.num_workers
    )

    results = filter(  
        model,
        loader,
        args.device,
        args.inference_label,
        tokenizer=tokenizer,
        output_csv=args.filter_output
    )

    return results
