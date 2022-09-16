#!/bin/bash
MODEL_PATH='./retriever_model_s1' # './co-condenser-marco' # './co-condenser-marco-retriever'
ENCODING_DIR='./encoding_s1' # 'encoding_s0' # './encoding'
NPROC=1

mkdir -p $ENCODING_DIR/corpus
mkdir -p $ENCODING_DIR/query

for i in $(seq -f "%02g" 0 9)
do
python -m torch.distributed.launch --nproc_per_node $NPROC -m tevatron.driver.encode \
 --output_dir ./retriever_model \
 --model_name_or_path $MODEL_PATH \
 --fp16 \
 --per_device_eval_batch_size 256 \
 --encode_in_path marco/bert/corpus/split${i}.json \
 --encoded_save_path $ENCODING_DIR/corpus/split${i}.pt
done


# python -m torch.distributed.launch --nproc_per_node $NPROC -m tevatron.driver.encode \
#  --output_dir ./retriever_model \
#  --model_name_or_path $MODEL_PATH \
#  --fp16 \
#  --q_max_len 32 \
#  --encode_is_qry \
#  --per_device_eval_batch_size 128 \
#  --encode_in_path marco/bert/query/dev.query.json \
#  --encoded_save_path $ENCODING_DIR/query/qry.pt
