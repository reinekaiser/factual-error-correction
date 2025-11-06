import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import  get_linear_schedule_with_warmup
import json
import logging
from .evaluate import evaluate_dev
from .helper import get_optimizer
from .dataset import Seq2SeqDataset
from tqdm import tqdm
# try:
#     from torch.utils.tensorboard import SummaryWriter
# except ImportError:
from tensorboardX import SummaryWriter


logger = logging.getLogger("__main__")


def train(model, tokenizer, args):
    """ Train the model """
    if args.warmup_ratio == 0 and args.warmup_steps == 0:
        logger.warning('You are training a model without using warmup.')
    elif args.warmup_ratio>0 and args.warmup_steps>0:
        raise ValueError("You can only specify either warmup_ratio or warmup_steps.")
    elif args.warmup_ratio>0:
        args.warmup_steps = int(args.warmup_ratio*args.max_steps)
        logger.info(f'warmup_steps is {args.warmup_steps}.')
    else:
        pass
    
    tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)

    optimizer = get_optimizer(args.optimizer, model, weight_decay=args.weight_decay, lr=args.lr, adam_epsilon=args.adam_epsilon)
   
    logger.info("  Max steps = %d", args.max_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_device_train_batch_size)
    logger.info("  Total train batch size = %d",
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    total_loss = 0.0
    model.zero_grad()
    model.train()

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )
    # optimizer = AdamW(model.parameters(), lr=args.lr)  # the learning rate is linearly scales with the #gpu
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=0, verbose=True, min_lr=1e-6)
    
    global_step = 0
    iter_count = 0

    train_dataset = Seq2SeqDataset(args.train_file, tokenizer,
                                   max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len,
                                   use_evidence=args.use_evidence,
                                   source_prefix = args.source_prefix, dataset_percent = args.dataset_percent, num_data_instance = args.num_data_instance,
                                   inference=False, 
                                   mask_ratio=args.mask_ratio, mask_strategy=args.mask_strategy, merge_mask=args.merge_mask, initialization=args.initialization
                                   )
    
    train_sampler = RandomSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  collate_fn=train_dataset.create_mini_batch,
                                  batch_size=args.per_device_train_batch_size, 
                                  num_workers=args.preprocessing_num_workers)
    if args.validation_file is not None:
        dev_dataset = Seq2SeqDataset(args.validation_file, tokenizer,
                                     max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len,
                                     use_evidence=args.use_evidence, 
                                     source_prefix = args.source_prefix, inference=False,
                                     mask_ratio=args.mask_ratio, mask_strategy=args.mask_strategy, merge_mask=args.merge_mask, initialization=args.initialization
                                    )
        dev_sampler = SequentialSampler(dev_dataset)
        dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler,
                                    collate_fn=dev_dataset.create_mini_batch,
                                    batch_size=args.per_device_eval_batch_size, 
                                    num_workers=args.preprocessing_num_workers)

        dev_loss = evaluate_dev(args, model, dev_dataloader)
        best_dev_loss = dev_loss
    trigger_times = 0
    while global_step < args.max_steps:
        iter_count += 1
        if args.num_train_epochs >0 and iter_count > args.num_train_epochs:
            break
        if trigger_times >= args.patience:
            logger.info('Early stopping!')
            break
        epoch_iterator = tqdm(train_dataloader, desc=f"Iteration {iter_count}")

        for step, batch_generator in enumerate(epoch_iterator):
            for k, v in batch_generator.items():
                batch_generator[k] = v.to(args.device)

            outputs = model(**batch_generator)
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss

            loss = loss / args.gradient_accumulation_steps
            total_loss += loss.item()
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    loss_scalar = total_loss / args.logging_steps
                    learning_rate_scalar = scheduler.get_last_lr()[0]
                    logs = {}
                    logs["train_learning_rate"] = learning_rate_scalar
                    logs["train_nll_loss"] = loss_scalar
                    total_loss = 0
                    
                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    logger.info(json.dumps({**logs, **{"step": global_step}}))
                    
                if args.save_steps > 0 and global_step % args.save_steps == 0 and args.validation_file is not None:
                    dev_loss = evaluate_dev(args, model, dev_dataloader)
                    model.train()
                    
                    tb_writer.add_scalar("dev_nll_loss", dev_loss, global_step)
                    if dev_loss<best_dev_loss:
                        logger.info('Save the model at {}.'.format(args.output_dir))
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(args.output_dir)
                        tokenizer.save_pretrained(args.output_dir)
                        trigger_times = 0
                        best_dev_loss = dev_loss
                    else:
                        trigger_times += 1
                        logger.info(f'Trigger times: {trigger_times}.')

            if global_step >= args.max_steps:
                break
            if trigger_times >= args.patience:
                logger.info('Early stopping!')
                break
                
    if args.validation_file is None:
        logger.info('Save the model at {}.'.format(args.output_dir))
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
    tb_writer.close()
    
    return global_step