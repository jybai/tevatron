#!/bin/bash
# remember to set CUDA_VISIBLE_DEVICES and MASTER_PORT
MODEL_PATH=$1
ENCODING_DIR=$MODEL_PATH'/embs'
BSIZE=4096

set -e

mkdir -p $ENCODING_DIR/corpus
mkdir -p $ENCODING_DIR/query

CUDA_DEVICE_LIST=(${CUDA_VISIBLE_DEVICES//,/ })

parallel --xapply -j ${#CUDA_DEVICE_LIST[@]} \
 CUDA_VISIBLE_DEVICES={2} \
 python -m torch.distributed.launch \
 --nproc_per_node 1 \
 --master_port=$MASTER_PORT{%} \
 -m tevatron.driver.encode \
 --output_dir ./retriever_model \
 --model_name_or_path $MODEL_PATH \
 --fp16 \
 --per_device_eval_batch_size $BSIZE \
 --encode_in_path marco/bert/corpus/split{1}.json \
 --encoded_save_path $ENCODING_DIR/corpus/split{1}.pt \
 ::: $(seq -f "%02g" 0 9) ::: ${CUDA_DEVICE_LIST[@]}

python -m torch.distributed.launch \
 --nproc_per_node 1 \
 --master_port=$MASTER_PORT \
 -m tevatron.driver.encode \
 --output_dir ./retriever_model \
 --model_name_or_path $MODEL_PATH \
 --fp16 \
 --encode_is_qry \
 --per_device_eval_batch_size $BSIZE \
 --encode_in_path marco/bert/query/train.query.json \
 --encoded_save_path $ENCODING_DIR/query/qry.trn.pt

python -m torch.distributed.launch \
 --nproc_per_node 1 \
 --master_port=$MASTER_PORT \
 -m tevatron.driver.encode \
 --output_dir ./retriever_model \
 --model_name_or_path $MODEL_PATH \
 --fp16 \
 --encode_is_qry \
 --per_device_eval_batch_size $BSIZE \
 --encode_in_path marco/bert/query/dev.query.json \
 --encoded_save_path $ENCODING_DIR/query/qry.dev.pt
