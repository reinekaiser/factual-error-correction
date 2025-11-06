import torch
from tqdm import tqdm

import logging
from .dataset import Seq2SeqDataset
from torch.utils.data import DataLoader, SequentialSampler
logger = logging.getLogger("__main__")

def evaluate_dev(args, model, dataloader):
    """
    compute the average loss over the test or validation set.
    :param args:
    :param model:
    :param dataloader:
    :return:
    """
    datasize = len(dataloader.dataset)
    model.eval()
    total_lm_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch_generator in tqdm(dataloader, desc="Evaluate"):
            for k, v in batch_generator.items():
                batch_generator[k] = v.to(args.device)
            outputs = model(**batch_generator)
            decoder_labels = batch_generator['labels']
            
            lm_loss = outputs.loss
            bts = decoder_labels.shape[0]
            num_tokens = torch.sum(decoder_labels != -100)
            total_lm_loss += lm_loss * num_tokens
            total_tokens += num_tokens


        average_lm_loss = total_lm_loss / total_tokens
        ave_loss = average_lm_loss.item()
    model.train()
    logger.info('Validation loss = %.3f.', ave_loss)
    return ave_loss
    
def evaluate(model, tokenizer, args):
    model.eval()
    dev_dataset = Seq2SeqDataset(args.validation_file, tokenizer,
                                max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len,
                                use_evidence=args.use_evidence, source_prefix = args.source_prefix, 
                                inference=False, 
                                mask_ratio=args.mask_ratio, mask_strategy=args.mask_strategy, merge_mask=args.merge_mask, initialization=args.initialization
                                )

    
    dev_sampler = SequentialSampler(dev_dataset)
    
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler,
                                collate_fn=dev_dataset.create_mini_batch,
                                batch_size=args.per_device_eval_batch_size, 
                                num_workers=args.preprocessing_num_workers)

    dev_loss = evaluate_dev(args, model, dev_dataloader)
    return dev_loss