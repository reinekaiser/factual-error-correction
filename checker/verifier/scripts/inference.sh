DATA_NAME=./checker/verifier/dummy
DATA_DIR=./$DATA_NAME
MODEL_PATH=$1
OUTPUT_PATH=$2

export PYTHONPATH=$(pwd)

python ./checker/verifier/main.py \
    --dev_file $DATA_DIR/test_beam_1.txt \
    --model_name $MODEL_PATH \
    --filter_output $OUTPUT_PATH \
    --resume \
    --do_inference