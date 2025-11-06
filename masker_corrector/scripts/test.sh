DATA_DIR=./masker_corrector/data
MODEL_PATH=$1

export PYTHONPATH=$(pwd)

python masker_corrector/main.py  \
    --test_file $DATA_DIR/test.csv \
    --do_predict \
    --model_path $MODEL_PATH --resume \
    --num_beams 1 \
    --mask_strategy heuristic