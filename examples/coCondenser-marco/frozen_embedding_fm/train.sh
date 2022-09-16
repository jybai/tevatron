#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py ../data/MS-MARCO/original/ --k 2 --auto_stop --save_model_dir /efs/core-pecos/users/cybai/pytorch-fm/models/fm_k2
