#!/bin/bash

python -m tevatron.faiss_retriever \
 --query_reps encoding/query/qry.pt \
 --passage_reps encoding/corpus/'*.pt' \
 --depth 10 \
 --batch_size -1 \
 --save_text \
 --save_ranking_to rank.tsv
