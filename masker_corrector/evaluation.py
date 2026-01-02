from evaluate import load
from bert_score import score as bert_score
import argparse
import os
import csv
from underthesea import word_tokenize
from tqdm import tqdm
import torch
import numpy as np

from collections import Counter

def SARIngram(sgrams, cgrams, rgramslist, numref):
    rgramsall = [rgram for rgrams in rgramslist for rgram in rgrams]
    rgramcounter = Counter(rgramsall)

    sgramcounter = Counter(sgrams)
    sgramcounter_rep = Counter()
    for sgram, scount in sgramcounter.items():
        sgramcounter_rep[sgram] = scount * numref

    cgramcounter = Counter(cgrams)
    cgramcounter_rep = Counter()
    for cgram, ccount in cgramcounter.items():
        cgramcounter_rep[cgram] = ccount * numref

    # KEEP
    keepgramcounter_rep = sgramcounter_rep & cgramcounter_rep
    keepgramcountergood_rep = keepgramcounter_rep & rgramcounter
    keepgramcounterall_rep = sgramcounter_rep & rgramcounter

    keeptmpscore1 = 0
    keeptmpscore2 = 0
    for keepgram in keepgramcountergood_rep:
        keeptmpscore1 += (
            keepgramcountergood_rep[keepgram] / keepgramcounter_rep[keepgram]
        )
        keeptmpscore2 += (
            keepgramcountergood_rep[keepgram] / keepgramcounterall_rep[keepgram]
        )
        # print "KEEP", keepgram, keepscore, cgramcounter[keepgram], sgramcounter[keepgram], rgramcounter[keepgram]
    keepscore_precision = 0
    if len(keepgramcounter_rep) > 0:
        keepscore_precision = keeptmpscore1 / len(keepgramcounter_rep)
    keepscore_recall = 0
    if len(keepgramcounterall_rep) > 0:
        keepscore_recall = keeptmpscore2 / len(keepgramcounterall_rep)
    keepscore = 0
    if keepscore_precision > 0 or keepscore_recall > 0:
        keepscore = (
            2
            * keepscore_precision
            * keepscore_recall
            / (keepscore_precision + keepscore_recall)
        )

    # DELETION
    delgramcounter_rep = sgramcounter_rep - cgramcounter_rep
    delgramcountergood_rep = delgramcounter_rep - rgramcounter
    delgramcounterall_rep = sgramcounter_rep - rgramcounter
    deltmpscore1 = 0
    deltmpscore2 = 0
    for delgram in delgramcountergood_rep:
        deltmpscore1 += delgramcountergood_rep[delgram] / delgramcounter_rep[delgram]
        deltmpscore2 += delgramcountergood_rep[delgram] / delgramcounterall_rep[delgram]
    delscore_precision = 0
    if len(delgramcounter_rep) > 0:
        delscore_precision = deltmpscore1 / len(delgramcounter_rep)
    delscore_recall = 0
    if len(delgramcounterall_rep) > 0:
        delscore_recall = deltmpscore1 / len(delgramcounterall_rep)
    delscore = 0
    if delscore_precision > 0 or delscore_recall > 0:
        delscore = (
            2
            * delscore_precision
            * delscore_recall
            / (delscore_precision + delscore_recall)
        )

    # ADDITION
    addgramcounter = set(cgramcounter) - set(sgramcounter)
    addgramcountergood = set(addgramcounter) & set(rgramcounter)
    addgramcounterall = set(rgramcounter) - set(sgramcounter)

    addtmpscore = 0
    for addgram in addgramcountergood:
        addtmpscore += 1

    addscore_precision = 0
    addscore_recall = 0
    if len(addgramcounter) > 0:
        addscore_precision = addtmpscore / len(addgramcounter)
    if len(addgramcounterall) > 0:
        addscore_recall = addtmpscore / len(addgramcounterall)
    addscore = 0
    if addscore_precision > 0 or addscore_recall > 0:
        addscore = (
            2
            * addscore_precision
            * addscore_recall
            / (addscore_precision + addscore_recall)
        )

    return (keepscore, delscore_precision, addscore)


def SARIsent(ssent, csent, rsents):
    numref = len(rsents)

    s1grams = ssent.lower().split(" ")
    c1grams = csent.lower().split(" ")
    s2grams = []
    c2grams = []
    s3grams = []
    c3grams = []
    s4grams = []
    c4grams = []

    r1gramslist = []
    r2gramslist = []
    r3gramslist = []
    r4gramslist = []
    for rsent in rsents:
        r1grams = rsent.lower().split(" ")
        r2grams = []
        r3grams = []
        r4grams = []
        r1gramslist.append(r1grams)
        for i in range(0, len(r1grams) - 1):
            if i < len(r1grams) - 1:
                r2gram = r1grams[i] + " " + r1grams[i + 1]
                r2grams.append(r2gram)
            if i < len(r1grams) - 2:
                r3gram = r1grams[i] + " " + r1grams[i + 1] + " " + r1grams[i + 2]
                r3grams.append(r3gram)
            if i < len(r1grams) - 3:
                r4gram = (
                    r1grams[i]
                    + " "
                    + r1grams[i + 1]
                    + " "
                    + r1grams[i + 2]
                    + " "
                    + r1grams[i + 3]
                )
                r4grams.append(r4gram)
        r2gramslist.append(r2grams)
        r3gramslist.append(r3grams)
        r4gramslist.append(r4grams)

    for i in range(0, len(s1grams) - 1):
        if i < len(s1grams) - 1:
            s2gram = s1grams[i] + " " + s1grams[i + 1]
            s2grams.append(s2gram)
        if i < len(s1grams) - 2:
            s3gram = s1grams[i] + " " + s1grams[i + 1] + " " + s1grams[i + 2]
            s3grams.append(s3gram)
        if i < len(s1grams) - 3:
            s4gram = (
                s1grams[i]
                + " "
                + s1grams[i + 1]
                + " "
                + s1grams[i + 2]
                + " "
                + s1grams[i + 3]
            )
            s4grams.append(s4gram)

    for i in range(0, len(c1grams) - 1):
        if i < len(c1grams) - 1:
            c2gram = c1grams[i] + " " + c1grams[i + 1]
            c2grams.append(c2gram)
        if i < len(c1grams) - 2:
            c3gram = c1grams[i] + " " + c1grams[i + 1] + " " + c1grams[i + 2]
            c3grams.append(c3gram)
        if i < len(c1grams) - 3:
            c4gram = (
                c1grams[i]
                + " "
                + c1grams[i + 1]
                + " "
                + c1grams[i + 2]
                + " "
                + c1grams[i + 3]
            )
            c4grams.append(c4gram)

    (keep1score, del1score, add1score) = SARIngram(
        s1grams, c1grams, r1gramslist, numref
    )
    (keep2score, del2score, add2score) = SARIngram(
        s2grams, c2grams, r2gramslist, numref
    )
    (keep3score, del3score, add3score) = SARIngram(
        s3grams, c3grams, r3gramslist, numref
    )
    (keep4score, del4score, add4score) = SARIngram(
        s4grams, c4grams, r4gramslist, numref
    )
    avgkeepscore = sum([keep1score, keep2score, keep3score, keep4score]) / 4
    avgdelscore = sum([del1score, del2score, del3score, del4score]) / 4
    avgaddscore = sum([add1score, add2score, add3score, add4score]) / 4
    finalscore = (avgkeepscore + avgdelscore + avgaddscore) / 3

    return {
        'sari': finalscore,
        'keep_score': avgkeepscore,
        'addition_score': avgaddscore,
        'deletion_score': avgdelscore
    }

def vi_tokenize(text):
    if not isinstance(text, str):
        return ""
    return word_tokenize(text.lower().strip(), format="text")
    
def load_data(filename, use_partial, label, tokenize_vi=True):
    sources = []
    predictions = []
    references = []
    
    with open(filename, mode='r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in csv.DictReader(f))
    
    with open(filename, mode='r', encoding='utf-8') as fr:
        reader = csv.DictReader(fr)
        for data_instance in tqdm(reader, total=total_lines, desc="Loading data"):
            if use_partial:
                if int(data_instance['labels']) != label:
                    continue
                
            src = data_instance['Statement']
            if int(data_instance['labels']) == 0:
                tgt = data_instance['Statement']
            else:
                tgt = data_instance['Mutated']
            gen = data_instance['generated_text']
            
            if isinstance(gen, list):
                gen = gen[0]
            elif isinstance(gen, str) and gen.startswith('[') and gen.endswith(']'):
                try:
                    import ast
                    gen = ast.literal_eval(gen)[0]
                except Exception:
                    pass  
            
            if tokenize_vi:
                src = vi_tokenize(src)
                tgt = vi_tokenize(tgt)
                gen = vi_tokenize(gen)
            
            sources.append(src)
            predictions.append(gen)
            references.append([tgt]) 
    
    return sources, predictions, references

def calculate_sari(sources, predictions, references):
    """
    Tính SARI cho corpus sử dụng numpy
    """
    scores = {
        'sari': [],
        'keep_score': [],
        'addition_score': [],
        'deletion_score': []
    }
    
    for ssent, psent, rsents in zip(sources, predictions, references):
        result = SARIsent(ssent, psent, rsents)
        
        for key in scores:
            scores[key].append(result[key])
    
    # Tính statistics
    return {
        'sari': np.mean(scores['sari']),
        'keep_score': np.mean(scores['keep_score']),
        'addition_score': np.mean(scores['addition_score']),
        'deletion_score': np.mean(scores['deletion_score'])
    }

def calculate_bertscore(predictions, references, model_type="vinai/phobert-base", batch_size=32):
    flat_references = [ref[0] if isinstance(ref, list) else ref for ref in references]

    model_layers = {
        'vinai/phobert-base': 12,
        'vinai/phobert-large': 24,
        'xlm-roberta-base': 12,
        'xlm-roberta-large': 24,
        'bert-base-multilingual-cased': 12,
    }
    
    num_layers = model_layers.get(model_type, 9)
    
    P, R, F1 = bert_score(
        predictions,
        flat_references,
        model_type=model_type,
        num_layers=num_layers,
        lang="vi",
        verbose=True,
        batch_size=batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item(),
        'precision_std': P.std().item(),
        'recall_std': R.std().item(),
        'f1_std': F1.std().item()
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factual Error Correction Evaluation.")
    parser.add_argument('--input_file', type=str, required=True,
                        help='The input CSV file for evaluation.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='The output file to save evaluation results.')
    parser.add_argument('--bert_model', type=str, default='xlm-roberta-base',
                        choices=['vinai/phobert-base', 'vinai/phobert-large', 
                                'xlm-roberta-base', 'xlm-roberta-large',
                                'bert-base-multilingual-cased'],
                        help='BERT model for BERTScore')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for BERTScore calculation')
    parser.add_argument('--label', type=int, default=1,
                        help='Lable')
    parser.add_argument('--use_partial', type=int, default=0)

    args = parser.parse_args()
    
    # Load metrics
    print("Loading metrics...")
    rouge = load('rouge')
    
    # Load data
    print(f'\nEvaluating {args.input_file}')
    sources, predictions, references = load_data(args.input_file, args.use_partial, args.label)
    print(f"Loaded {len(sources)} examples\n")
    
    # Tạo header nếu file chưa tồn tại
    file_exists = os.path.exists(args.output_file)
    with open(args.output_file, 'a', encoding='utf-8') as fw:
        if not file_exists:
            fw.write("SARI,SARI-Keep,SARI-Add,SARI-Del,Rouge2,BERTScore-P,BERTScore-R,BERTScore-F1,Filename\n")
            

        # Tính SARI với chi tiết
        print("="*60)
        print("Calculating SARI...")
        results_sari = calculate_sari(sources, predictions, references)
        
        print(f"✓ SARI Overall: {results_sari['sari']:.3f}")
        print(f"  - Keep Score:     {results_sari['keep_score']:.3f}")
        print(f"  - Addition Score: {results_sari['addition_score']:.3f}")
        print(f"  - Deletion Score: {results_sari['deletion_score']:.3f}")
        
        # Tính ROUGE
        print("\nCalculating ROUGE...")
        results_rouge = rouge.compute(predictions=predictions, references=references)
        print(f"✓ ROUGE-2: {100*results_rouge['rouge2']:.2f}")
        
        # Tính BERTScore
        print(f"\nCalculating BERTScore (model: {args.bert_model})...")
        results_bert = calculate_bertscore(
            predictions, 
            references, 
            model_type=args.bert_model,
            batch_size=args.batch_size
        )
        print(f"✓ BERTScore:")
        print(f"  - Precision: {100*results_bert['precision']:.2f} ± {100*results_bert['precision_std']:.2f}")
        print(f"  - Recall:    {100*results_bert['recall']:.2f} ± {100*results_bert['recall_std']:.2f}")
        print(f"  - F1:        {100*results_bert['f1']:.2f} ± {100*results_bert['f1_std']:.2f}")
        

        output = (f"{results_sari['sari']:.2f},"
                 f"{results_sari['keep_score']:.2f},"
                 f"{results_sari['addition_score']:.2f},"
                 f"{results_sari['deletion_score']:.2f},"
                 f"{100*results_rouge['rouge2']:.2f},"
                 f"{100*results_bert['precision']:.2f},"
                 f"{100*results_bert['recall']:.2f},"
                 f"{100*results_bert['f1']:.2f},"
                 f"{args.input_file}\n")
        
        fw.write(output)
        
        print("="*60)
        print(f"\n Results saved to {args.output_file}")