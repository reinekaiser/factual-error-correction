from tqdm import tqdm
import torch
import pandas as pd
import os
import re
from .cr_dataset import CRDataset
from .helper import collate_fn
from functools import partial
from torch.utils.data import DataLoader, SequentialSampler

def reconstruct(masked_input, generated_text, tokenizer):
    """
    Merge generated_text vào masked_input đúng vị trí <extra_id_n>
    """
    input_tokens = tokenizer.tokenize(masked_input)
    gen_tokens = tokenizer.tokenize(generated_text)

    result_tokens = []
    gen_idx = 0

    i = 0
    while i < len(input_tokens):
        token = input_tokens[i]
        if token.startswith("<extra_id_"):
            fill_tokens = []
            while gen_idx < len(gen_tokens) and not gen_tokens[gen_idx].startswith("<extra_id_"):
                fill_tokens.append(gen_tokens[gen_idx])
                gen_idx += 1
            result_tokens.extend(fill_tokens)
            i += 1
        else:
            result_tokens.append(token)
            i += 1

    return tokenizer.convert_tokens_to_string(result_tokens)

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

            gen_texts = [reconstruct(src, gen, tokenizer) for src, gen in zip(src_texts, raw_gen_texts)]

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

