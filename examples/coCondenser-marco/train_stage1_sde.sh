#!/bin/bash

# remember to set CUDA_VISIBLE_DEVICES and MASTER_PORT

CONDENSER_MODEL_NAME="./co-condenser-marco"
MODEL_PATH='./retriever_model_s1_sde'

python -m torch.distributed.launch --nproc_per_node=4 \
	--master_port=$MASTER_PORT -m tevatron.driver.train \
	--output_dir $MODEL_PATH \
	--add_pooler \
	--model_name_or_path $CONDENSER_MODEL_NAME \
	--save_steps 20000 \
	--train_dir ./marco/bert/train \
	--fp16 \
	--per_device_train_batch_size 2 \
	--train_n_passages 8 \
	--learning_rate 5e-6 \
	--num_train_epochs 3 \
	--dataloader_num_workers 4 \
	--negatives_x_device

./encode_general.sh $MODEL_PATH
