#!/bin/bash
ENCODING_DIR=$1
LABEL_DIR="../data/MS-MARCO/Ys/"

python train.py $ENCODING_DIR $LABEL_DIR --k 768 --Y_trn_suffix 400k --eval_zeroth_step --identity_factor None --nepochs 0 --eval_trn --ip
