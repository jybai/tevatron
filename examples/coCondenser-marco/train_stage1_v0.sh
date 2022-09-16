#!/bin/bash

CONDENSER_MODEL_NAME="./co-condenser-marco"
MODEL_PATH='./retriever_model_s1_v0' # './co-condenser-marco' # './co-condenser-marco-retriever'
ENCODING_DIR=$MODEL_PATH'/embs' # 'encoding_s0' # './encoding'
PORT=1127
BSIZE=4096

python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT -m tevatron.driver.train \
	--output_dir $MODEL_PATH \
	--model_name_or_path $CONDENSER_MODEL_NAME \
	--save_steps 20000 \
	--train_dir ./marco/bert/train \
	--fp16 \
	--per_device_train_batch_size 8 \
	--learning_rate 5e-6 \
	--num_train_epochs 3 \
	--dataloader_num_workers 2

mkdir -p $ENCODING_DIR/corpus
mkdir -p $ENCODING_DIR/query

python -m torch.distributed.launch --nproc_per_node 1 --master_port=$PORT -m tevatron.driver.encode \
 --output_dir ./retriever_model \
 --model_name_or_path $MODEL_PATH \
 --fp16 \
 --encode_is_qry \
 --per_device_eval_batch_size $BSIZE \
 --encode_in_path marco/bert/query/train.query.json \
 --encoded_save_path $ENCODING_DIR/query/qry.trn.pt

python -m torch.distributed.launch --nproc_per_node 1 --master_port=$PORT -m tevatron.driver.encode \
 --output_dir ./retriever_model \
 --model_name_or_path $MODEL_PATH \
 --fp16 \
 --encode_is_qry \
 --per_device_eval_batch_size $BSIZE \
 --encode_in_path marco/bert/query/dev.query.json \
 --encoded_save_path $ENCODING_DIR/query/qry.dev.pt

for i in $(seq -f "%02g" 0 9)
do
python -m torch.distributed.launch --nproc_per_node 1 -m tevatron.driver.encode \
 --output_dir ./retriever_model \
 --model_name_or_path $MODEL_PATH \
 --fp16 \
 --per_device_eval_batch_size $BSIZE \
 --encode_in_path marco/bert/corpus/split${i}.json \
 --encoded_save_path $ENCODING_DIR/corpus/split${i}.pt
done
