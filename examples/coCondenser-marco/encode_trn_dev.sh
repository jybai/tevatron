#!/bin/bash
MODEL_PATH='./retriever_model_s1' # './co-condenser-marco' # './co-condenser-marco-retriever'
ENCODING_DIR='./encoding_s1' # 'encoding_s0' # './encoding'
NPROC=1
PORT=1126
BSIZE=256

mkdir -p $ENCODING_DIR/query

python -m torch.distributed.launch --nproc_per_node $NPROC --master_port=$PORT -m tevatron.driver.encode \
 --output_dir ./retriever_model \
 --model_name_or_path $MODEL_PATH \
 --fp16 \
 --q_max_len 32 \
 --encode_is_qry \
 --per_device_eval_batch_size $BSIZE \
 --encode_in_path marco/bert/query/train.query.json \
 --encoded_save_path $ENCODING_DIR/query/qry.trn.pt

python -m torch.distributed.launch --nproc_per_node $NPROC --master_port=$PORT -m tevatron.driver.encode \
 --output_dir ./retriever_model \
 --model_name_or_path $MODEL_PATH \
 --fp16 \
 --q_max_len 32 \
 --encode_is_qry \
 --per_device_eval_batch_size $BSIZE \
 --encode_in_path marco/bert/query/dev.query.json \
 --encoded_save_path $ENCODING_DIR/query/qry.dev.pt
