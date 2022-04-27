## Train SPLADE on MS MARCO
```bash
CUDA_VISIBLE_DEVICES=0 python examples/splade/train_splade.py \
  --output_dir model_msmarco_splade \
  --model_name_or_path bert-base-uncased \
  --save_steps 20000 \
  --dataset_name Tevatron/msmarco-passage \
  --fp16 \
  --per_device_train_batch_size 8 \
  --train_n_passages 8 \
  --learning_rate 5e-6 \
  --q_max_len 16 \
  --p_max_len 128 \
  --q_flops_loss_factor 0.1 \
  --p_flops_loss_factor 0.1 \
  --num_train_epochs 3 \
  --logging_steps 500 \
  --overwrite_output_dir
```