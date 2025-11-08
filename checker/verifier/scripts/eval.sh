DATA_NAME=./checker/verifier/dummy
DATA_DIR=./$DATA_NAME
MODEL_PATH=$1

export PYTHONPATH=$(pwd)

python ./checker/verifier/main.py \
    --dev_file $DATA_DIR/test.parquet \
    --model_name $MODEL_PATH \
    --resume \
    --do_eval