import argparse
import logging

from utils.evaluate import evaluate
from utils.predict import predict
from utils.train import train
from utils.helper import load_model, set_env, str2bool


logger = logging.getLogger("__main__")

def get_parameter():
    parser = argparse.ArgumentParser(description="Factual Error Correction.")
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_eval', action='store_true', help='Whether to run eval on the dev set.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predictions on the test set.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='seq2seq_data',
                        help='the path of the src and tgt data.')
    parser.add_argument('--train_file', type=str, default='../data/train.csv',
                        help='The input training data file (a jsonlines).')
    parser.add_argument('--validation_file', type=str, default=None,
                        help='An optional input evaluation data file to evaluate the metrics (nll loss) on a jsonlines file.')  
    parser.add_argument('--test_file', type=str, default='../data/test.csv',
                        help='An optional input test data file to evaluate the metrics (sari) on a jsonlines file.')  
    
    parser.add_argument('--dataset_percent', type=float, default=1,
                        help='The percentage of data used to train the model.')
    parser.add_argument('--num_data_instance', type=int, default=-1,
                        help='The number of data instances used to train the model. -1 denotes using all data.')

    parser.add_argument('--model_path', type=str, default='',
                        help='Path to pretrained model or model identifier from huggingface.co/models')
          
    parser.add_argument('--initialization', type=str, default='VietAI/vit5-base',
                        choices=["google/byt5-small", 
                                 "VietAI/vit5-base", 
                                 "vinai/bartpho-syllable-base",         
                                 "vinai/bartpho-word-base"
                                 ],
                        help='initialize the model with random values byt5, vit5 or bartpho.')
    # hyper-paramters for training
    parser.add_argument('--per_device_train_batch_size', type=int, default=16)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help= "Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=-1,
                        help='If > 0: set total number of training steps to perform. Override num_train_epochs.')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Linear warmup over warmup_steps.')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Linear warmup over warmup_ratio fraction of total steps.')
    
    parser.add_argument('--optimizer', type=str, default='adamW', help='The optimizer to use.')
    parser.add_argument('--lr', type=float, default=4e-5, help='The initial learning rate for training.')
    # adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    # adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for AdamW if we apply some.')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Epsilon for AdamW optimizer.')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm.')
    
    parser.add_argument('--patience', type=int, default=4,
                        help='If the performance of model on the validation does not improve for n times, we will stop training.')

    parser.add_argument('--resume', action='store_true', help='whether load the best checkpoint or not.')
    # parameters for models
    parser.add_argument("--source_prefix", type=str, default=None, help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument('--max_src_len', type=int, default=256, help='the max length of the source text.')
    parser.add_argument('--max_tgt_len', type=int, default=256, help='the max length of the tgt text.')

    parser.add_argument('--use_evidence', type=str2bool, default=True, help='whether use evidences to revise the original claim.')
    parser.add_argument('--use_gold_evidence', type=str2bool, default=True, help='whether use gold evidences to revise the original claim.')
    parser.add_argument('--num_evidence', type=int, default=3,
                        help='the number of evidences used to revise the original claim.')
    parser.add_argument('--mask_ratio', type=float, default=0.15, help='The mask ratio for the source claim.')
    parser.add_argument('--mask_strategy', type=str, default='random', choices=['random', 'heuristic'], help='The mask strategy for the source claim.')
    parser.add_argument('--merge_mask', type=str2bool, default=False, help='whether use one mask token to denote multiple masked tokens.')
    
    # paramters for log
    parser.add_argument('--logging_steps', type=int, default=100, help='Log every X updates steps.')
    parser.add_argument('--save_steps', type=int, default=100, help='Save checkpoint every X updates steps.')
    parser.add_argument('--tensorboard_dir', type=str, default="../tensorboard_log", help="Tensorboard log dir.")
    parser.add_argument("--output_dir", type=str, default=None, help="dir for model checkpoints, logs and generated text.")
    
    # parameters for decoding
    # Arguments will be passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.
    parser.add_argument('--num_beams', type=int, default=1, help='Number of beams for beam search. 1 means no beam search.')
    parser.add_argument('--do_sample', action='store_true', help='Whether or not to use sampling; use greedy/beam search decoding otherwise.')
    parser.add_argument('--top_k', type=int, default=0, help='The number of highest probability vocabulary tokens to keep for top-k-filtering.')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help="If set to float < 1, only the most probable tokens with probabilities "
                             "that add up to `top_p` or higher are kept for generation.")

    parser.add_argument('--num_beam_groups', type=int, default=1,
                        help="Number of groups to divide num_beams into in order to ensure diversity among different groups of beams."
                         "This paper (https://arxiv.org/pdf/1610.02424.pdf) for more details.")
    parser.add_argument('--num_return_sequences', type=int, default=1,
                        help="The number of independently computed returned sequences for each element in the batch.")
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help="The parameter for repetition penalty. 1.0 means no penalty. "
                        "See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.")
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="The value used to module the next token probabilities.")
    
    parser.add_argument('--preprocessing_num_workers', type=int, default=2,
                        help='The number of processes to use for the preprocessing.')

    args = parser.parse_args()
    assert args.do_train + args.do_eval + args.do_predict == 1, print('Specify do_train, do_eval or do_predict.')
    assert args.mask_ratio>0 and args.mask_ratio<=1

    dir_prefix = f"{args.initialization.replace('/','-')}/seed{args.seed}_lr{args.lr}"


    if args.merge_mask:
        dir_prefix += '_merge-mask'
    
    if args.dataset_percent<1:
        dir_prefix += f'_{args.dataset_percent}-data-percent'
    if args.num_data_instance>0:
        dir_prefix += f'_{args.num_data_instance}-data-instance'
    assert args.dataset_percent==1 or args.num_data_instance==-1, print("Do not set both dataset_percent and num_data_instance.")
    
    if args.output_dir is None:
        if args.do_train:
            args.output_dir = f'/kaggle/working/checkpoints_masker_corrector/{dir_prefix}'
        else:
            args.output_dir = args.model_path
    args.tensorboard_dir = f'/kaggle/working/tensorboard_log_masker_corrector/{dir_prefix}'
    args.log_file = f'{args.output_dir}/log.txt'
    
    return args

def main():
    args = get_parameter()
    set_env(args)
    tokenizer, model = load_model(args)
    
    if args.do_train:
        logger.info("*** Train ***")
        logger.info("args:\n%s", '\n'.join([f'    {arg}={getattr(args, arg)}'  for arg in vars(args)]))
        global_step = train(model, tokenizer, args)
        logger.info(" global_step = %s", global_step)

    if args.do_eval:
        logger.info("*** Evaluate ***") 
        evaluate(model, tokenizer, args)

    if args.do_predict:
        logger.info("*** Predict ***")
        print("args:\n%s", '\n'.join([f'    {arg}={getattr(args, arg)}'  for arg in vars(args)]))
        predict(model, tokenizer, args)
    

if __name__ == "__main__":
    main()