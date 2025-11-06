DATA_DIR=./masker_corrector/data
INIT_MODEL=${1:-VietAI/vit5-base}
export PYTHONPATH=$(pwd)

python masker_corrector/main.py  \
    --initialization $INIT_MODEL \
    --train_file $DATA_DIR/train.csv \
    --validation_file $DATA_DIR/dev.csv \
    --max_src_len 256 --max_tgt_len 256 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --lr 4e-5 --patience 2\
    --logging_steps 100 --save_steps 200 --max_steps 4000  \
    --do_train \
    --mask_strategy random --mask_ratio 0.5