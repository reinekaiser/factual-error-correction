from evaluate import load
import argparse
import os
import csv
from underthesea import word_tokenize

def vi_tokenize(text):
    if not isinstance(text, str):
        return ""
    return word_tokenize(text.lower().strip(), format="text")
    
def load_data(filename, tokenize_vi=True, source_column="Statement", target_column="Mutated", generated_column="generated_text"):
    sources = []
    predictions = []
    references = []

    with open(filename, mode='r', encoding='utf-8') as fr:
        reader = csv.DictReader(fr)
        for data_instance in reader:
            src = data_instance[source_column]
            tgt = data_instance[target_column]
            gen = data_instance[generated_column]

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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factual Error Correction.")
    parser.add_argument('--input_file', type=str, default='',
                        help='The input file for evaluation (a jsonlines).')
    parser.add_argument('--output_file', type=str,
                    help='The output file used to save the evaluation results.')
    
    args = parser.parse_args()

    sari = load("sari")
    rouge = load('rouge')

    print(f'Evaluate {args.input_file}.')
    sources, predictions, references = load_data(args.input_file)

    if not os.path.exists(args.output_file):
        with open(args.output_file, 'a') as fw:
            fw.write("SARI, Rouge2, Filename\n")
    
    with open(args.output_file, 'a') as fw:
        results = sari.compute(sources=sources, predictions=predictions, references=references)
        print(results)
        results2 = rouge.compute(predictions=predictions, references=references)
        print(results2)
        output = f"{results['sari']:.2f}, {100*results2['rouge2']:.2f}, {args.input_file}\n"
        fw.write(output)