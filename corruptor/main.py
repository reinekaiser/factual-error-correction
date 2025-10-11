import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from masker import Masker

def main(args):
    if args.data.endswith(".csv"):
        df = pd.read_csv(args.data)
    elif args.data.endswith(".json"):
        df = pd.read_json(args.data)
    elif args.data.endswith(".parquet"):
        df = pd.read_parquet(args.data)
    else:
        raise ValueError("Unsupported file format")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    masker = Masker(tokenizer_name=args.tokenizer_name, use_mlm_fill = True, mask_prob = args.mask_prob)

    masked_claims = []
    filled_claims = []

    for idx, row in df.iterrows():
        claim = row[args.claim_col]
        evidence = row[args.evidence_col]

        masked_claim, filled_claim = masker.tokenizer_mask_and_fill_wrong(claim, evidence, top_k=args.top_k)
        masked_claims.append(masked_claim)
        filled_claims.append(filled_claim)

    # Add corrupted claims and new labels to DataFrame
    df["Masked_Claim"] = masked_claims
    df["Corrupted_Claim"] = filled_claims

    # Save masked DataFrame
    output_file = args.output if args.output else "masked_df.csv"
    if output_file.endswith(".csv"):
        df.to_csv(output_file, index=False)
    elif output_file.endswith(".json"):
        df.to_json(output_file, orient="records", lines=True)
    elif output_file.endswith(".parquet"):
        df.to_parquet(output_file, index=False)
    else:
        raise ValueError("Unsupported output file format")

    print(f"Masked DataFrame saved to {output_file}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to CSV/JSON/Parquet dataset")
    parser.add_argument("--tokenizer_name", type=str, default="xlm-roberta-base")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--claim_col", type=str, default="Statement")
    parser.add_argument("--evidence_col", type=str, default="Evidence")
    parser.add_argument("--output", type=str)
    parser.add_argument("--label_col", type=str, default="labels")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--mask_prob", type=float, default=0.3)
    
    args = parser.parse_args()
    main(args)

