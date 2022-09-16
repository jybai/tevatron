#!/bin/bash

DEPTH=1000

# encode
python encode.py ../data/MS-MARCO/original/ /efs/core-pecos/users/cybai/pytorch-fm/models/$1 ../data/MS-MARCO/$1 --pickle

# retrieve
RETRIEVER='retrieve.py' # '-m tevatron.faiss_retriever'

python $RETRIEVER \
 --query_reps ../data/MS-MARCO/$1/X.trn.pkl \
 --passage_reps ../data/MS-MARCO/$1/Z.pkl \
 --depth $DEPTH \
 --batch_size -1 \
 --save_text \
 --save_ranking_to ../data/MS-MARCO/$1/rank.trn.tsv \
 --marco

python $RETRIEVER \
 --query_reps ../data/MS-MARCO/$1/X.tst.pkl \
 --passage_reps ../data/MS-MARCO/$1/Z.pkl \
 --depth $DEPTH \
 --batch_size -1 \
 --save_text \
 --save_ranking_to ../data/MS-MARCO/$1/rank.tst.tsv \
 --marco

# eval metrics
python ms_marco_eval.py /efs/core-pecos/users/cybai/tevatron/examples/coCondenser-marco/marco/qrels.train.tsv ../data/MS-MARCO/$1/rank.trn.tsv.marco
python ms_marco_eval.py /efs/core-pecos/users/cybai/tevatron/examples/coCondenser-marco/marco/qrels.dev.tsv ../data/MS-MARCO/$1/rank.tst.tsv.marco
