from tqdm import tqdm
import torch
import pandas as pd
import os
import re
from .cr_dataset import CRDataset
from .helper import collate_fn
from functools import partial
from torch.utils.data import DataLoader, SequentialSampler

def reconstruct(masked_input, generated_text):
    """
    Đố biết là cái gì
    """
    spans = re.split(r"<extra_id_\d+>", masked_input)
    preds = re.findall(r"<extra_id_\d+>([^<]*)", generated_text)
    
    result = []
    for i, span in enumerate(spans):
        result.append(span)
        if i < len(preds):
            result.append(preds[i].strip())
    return "".join(result).strip()

def generate(model, tokenizer, dataloader, device,
             generated_dir = "./generated.csv", 
             max_len = 128,
             num_beams = 4, do_sample = False,
             top_k = 50, top_p = 0.9, temperature = 1.0):
    model.eval()
    model.to(device)

    all_src, all_gen = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model.generate(
                input_ids = input_ids,
                attention_mask = attention_mask,
                max_length = max_len,
                num_beams = num_beams,
                do_sample = do_sample,
                top_k = top_k,
                top_p = top_p,
                temperature = temperature,
                no_repeat_ngram_size=3,
                repetition_penalty=1.8,
            )

            src_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            raw_gen_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)

            gen_texts = [reconstruct(src, gen) for src, gen in zip(src_texts, raw_gen_texts)]

            all_src.extend(src_texts)
            all_gen.extend(gen_texts)

    df = pd.DataFrame({
        "input_text": all_src,
        "generated_text": all_gen
    })

    if generated_dir.endswith(".xlsx"):
        df.to_excel(generated_dir, index=False)
    else:
        df.to_csv(generated_dir, index=False, encoding="utf-8")

    print(f"\nSaved generated results to: {generated_dir}")
    return df


def predict(model, tokenizer, args):
    test = CRDataset(
        args.test_dir,
        tokenizer = tokenizer,
        max_len = args.max_len,
        mask_ratio = args.mask_ratio,
        src_column = args.src_column,
        tgt_column = args.tgt_column,
        evidence_column = args.evidence_column,
        label_column = args.label_column,
        selected_label = args.selected_label,
        is_inference = True
    )

    test_loader = DataLoader(
        test,
        sampler = SequentialSampler(test),
        collate_fn = partial(collate_fn, tokenizer=tokenizer, inference = True),
        batch_size = args.batch_size,
        num_workers = args.num_workers
    )
    
    return generate(
        model, tokenizer, test_loader, args.device,
        generated_dir = args.generated_dir, 
        max_len = args.max_len,
        num_beams = args.num_beams, do_sample = args.do_sample,
        top_k = args.top_k, top_p = args.top_p, temperature = args.temperature
    )

