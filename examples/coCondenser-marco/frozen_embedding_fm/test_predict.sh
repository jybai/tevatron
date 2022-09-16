#!/bin/bash

# retrieve
SRC_DIR='../'$1'/embs' # '../data/MS-MARCO/ip'
RETRIEVER='retrieve.py' # '-m tevatron.faiss_retriever'

python $RETRIEVER \
 --query_reps $SRC_DIR/query/qry.trn.pt \
 --passage_reps $SRC_DIR/corpus/'*.pt' \
 --depth 10 \
 --batch_size -1 \
 --save_text \
 --save_ranking_to $SRC_DIR/rank.trn.tsv \
 --marco

python $RETRIEVER \
 --query_reps $SRC_DIR/query/qry.dev.pt \
 --passage_reps $SRC_DIR/corpus/'*.pt' \
 --depth 10 \
 --batch_size -1 \
 --save_text \
 --save_ranking_to $SRC_DIR/rank.tst.tsv \
 --marco

# eval metrics
python ms_marco_eval.py /efs/core-pecos/users/cybai/tevatron/examples/coCondenser-marco/marco/qrels.train.tsv $SRC_DIR/rank.trn.tsv.marco
python ms_marco_eval.py /efs/core-pecos/users/cybai/tevatron/examples/coCondenser-marco/marco/qrels.dev.tsv $SRC_DIR/rank.tst.tsv.marco

