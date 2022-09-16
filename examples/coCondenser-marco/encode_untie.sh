#!/bin/bash
MODEL_PATH='./retriever_model_s1_untie' # './co-condenser-marco' # './co-condenser-marco-retriever'
CONFIG_PATH='./retriever_model_s1_untie/query_model'
ENCODING_DIR='./encoding_s1_untie' # 'encoding_s0' # './encoding'
NPROC=1
PORT=1128
BSIZE=4096

mkdir -p $ENCODING_DIR/corpus
mkdir -p $ENCODING_DIR/query

python -m torch.distributed.launch --nproc_per_node $NPROC --master_port=$PORT -m tevatron.driver.encode \
 --output_dir ./retriever_model \
 --model_name_or_path $MODEL_PATH \
 --config_name $CONFIG_PATH \
 --fp16 \
 --untie_encoder \
 --encode_is_qry \
 --per_device_eval_batch_size $BSIZE \
 --encode_in_path marco/bert/query/train.query.json \
 --encoded_save_path $ENCODING_DIR/query/qry.trn.pt

python -m torch.distributed.launch --nproc_per_node $NPROC --master_port=$PORT -m tevatron.driver.encode \
 --output_dir ./retriever_model \
 --model_name_or_path $MODEL_PATH \
 --config_name $CONFIG_PATH \
 --fp16 \
 --untie_encoder \
 --encode_is_qry \
 --per_device_eval_batch_size $BSIZE \
 --encode_in_path marco/bert/query/dev.query.json \
 --encoded_save_path $ENCODING_DIR/query/qry.dev.pt

for i in $(seq -f "%02g" 0 9)
do
python -m torch.distributed.launch --nproc_per_node $NPROC -m tevatron.driver.encode \
 --output_dir ./retriever_model \
 --model_name_or_path $MODEL_PATH \
 --config_name $CONFIG_PATH \
 --fp16 \
 --untie_encoder \
 --per_device_eval_batch_size $BSIZE \
 --encode_in_path marco/bert/corpus/split${i}.json \
 --encoded_save_path $ENCODING_DIR/corpus/split${i}.pt
done
