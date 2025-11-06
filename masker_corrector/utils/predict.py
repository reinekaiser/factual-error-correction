import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import csv
import os

from .dataset import Seq2SeqDataset

def generate(model, tokenizer, test_dataloader, args):
    """
    Generate queries with the well-trained model.
    """
    prefix = os.path.basename(args.test_file).split('.')[0]

    if args.do_sample:
        if args.top_k>0:
            prefix += f'_top_k_{args.top_k}'
        else:
            assert args.top_p<1
            prefix += f'_top_p_{args.top_p}'
    else:
        prefix += f'_beam_{args.num_beams}'
    
    if args.num_return_sequences>1: 
        prefix += f'_rs_{args.num_return_sequences}'
    if args.repetition_penalty>1:
        prefix += f'_rp_{args.repetition_penalty}'
    if args.temperature!=1: 
        prefix += f'_tp_{args.temperature}'
    model.eval()
    
    output_filename = os.path.join(args.output_dir, prefix + ".csv")
   
    with open(output_filename, 'w', encoding='utf-8', newline='') as fw:
        writer = None

        with torch.no_grad():
            for batch_generator in tqdm(test_dataloader, desc="Generate"):
                outputs = model.generate(input_ids = batch_generator['input_ids'].to(args.device), 
                                     attention_mask = batch_generator['attention_mask'].to(args.device),
                                     max_length = args.max_tgt_len,
                                     num_beams = args.num_beams,
                                     do_sample  = args.do_sample,
                                     top_p = args.top_p,
                                     num_beam_groups = args.num_beam_groups,
                                     num_return_sequences = args.num_return_sequences,
                                     repetition_penalty = args.repetition_penalty,
                                     temperature = args.temperature)
                generated_example = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                assert len(generated_example)%args.num_return_sequences==0
                tmp_list = []
                example_list = []
                for i, example in enumerate(generated_example):
                    tmp_list.append(example)
                    if len(tmp_list) == args.num_return_sequences:
                        example_list.append(tmp_list)
                        tmp_list = []
                
                for data_instance, example in zip(batch_generator['data'], example_list):
                    instance = data_instance['original']
                    instance['masked_src'] = data_instance['masked_src']
                    instance['generated_text'] = example[0]
                    
                    if writer is None:
                        writer = csv.DictWriter(fw, fieldnames=list(instance.keys()))
                        writer.writeheader()

                    writer.writerow(instance)
            

def predict(model, tokenizer, args):
    model.eval()
    with open(args.test_file, 'r') as fr:
        num_data_instances = len(fr.readlines())

    start_idx = 0
    end_idx = num_data_instances
    
    test_dataset = Seq2SeqDataset(args.test_file, tokenizer,
                                    max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len,
                                    use_evidence=args.use_evidence, source_prefix = args.source_prefix, 
                                    inference=True, start_idx=start_idx, end_idx=end_idx,
                                    use_gold_evidence=args.use_gold_evidence, num_evidence=args.num_evidence,
                                    mask_ratio=args.mask_ratio, mask_strategy=args.mask_strategy,merge_mask=args.merge_mask,initialization=args.initialization)
    
    test_sampler = SequentialSampler(test_dataset)
    
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                 collate_fn=test_dataset.create_mini_batch,
                                 batch_size=args.per_device_eval_batch_size, 
                                 num_workers=args.preprocessing_num_workers)
    generate(model, tokenizer, test_dataloader, args)