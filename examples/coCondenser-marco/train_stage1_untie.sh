#!/bin/bash

CONDENSER_MODEL_NAME="./co-condenser-marco"

python -m torch.distributed.launch --nproc_per_node=4 --master_port=1128 -m tevatron.driver.train \
	--output_dir ./retriever_model_s1_untie \
	--untie_encoder \
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
