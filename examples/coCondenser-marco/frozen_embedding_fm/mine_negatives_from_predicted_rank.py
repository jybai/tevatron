import argparse
import os
import numpy as np
import pandas as pd
from scipy import sparse
import pickle
import glob
from multiprocessing import Pool
import random
from tqdm.auto import tqdm
from functools import partial

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_rank_path", help="Path to predicted rank on the training set, something like `train.rank.tsv`.", type=str)
    parser.add_argument("qrels_path", help="Path to ground truth relevant passages path of the training set (`qrels.train.tsv`).", type=str)
    parser.add_argument("save_Y_path", help="Path to save the hardmined Y.", type=str)
    
    parser.add_argument('--depth', type=int, default=200, help='Number of negatives to mine per query.')
    parser.add_argument('--n_sample', type=int, default=30, help='Number of actual negatives to sub-sample from the mined negatives.')
    parser.add_argument('--mp_chunk_size', type=int, default=100)
    parser.add_argument('--mp_threads', type=int, default=32)
    
    return parser.parse_args()

def mine_one(qid, depth, n_sample):
    pids_gt = qid_to_pids_gt[qid]
    pids_pred = qid_to_pids_pred[qid]
    
    assert(len(pids_gt) == len(pids_gt))
    assert(len(pids_pred) == len(set(pids_pred)))
    
    pids_pred_neg = []
    
    for pid_pred in pids_pred:
        if pid_pred not in pids_gt:
            pids_pred_neg.append(pid_pred)
        if len(pids_pred_neg) >= depth:
            break
            
    random.shuffle(pids_pred_neg)
    pids_pred_neg = pids_pred_neg[:n_sample]
    
    y_row = list(pids_gt) + pids_pred_neg
    y_col = [qid] * len(y_row)
    y_data = [1.0] * len(pids_gt) + [-1.0] * len(pids_pred_neg)
    
    perm = np.random.permutation(len(y_row))
    y_row, y_col, y_data = np.array(y_row)[perm], np.array(y_col)[perm], np.array(y_data)[perm]
    
    return np.stack([y_col, y_row, y_data], axis=1)
    
def main():
    args = parse_args()
    
    nx_trn = 502939 
    nx_tst = 6980 
    nz = 8841823
    
    Y_df = pd.read_csv(args.qrels_path, header=None, delimiter='\t')
    Y_df.drop(columns=[1, 3], inplace=True)
    Y_df.columns = ['query', 'passage']
    Y_df = Y_df.astype(int)
    
    Y_pred_df = pd.read_csv(args.pred_rank_path, header=None, delimiter='\t')
    Y_pred_df.drop(columns=[2], inplace=True)
    Y_pred_df.columns = ['query', 'passage']
    Y_pred_df = Y_pred_df.astype(int)
    
    assert(set(Y_df['query'].unique()) == set(Y_pred_df['query'].unique()))
    
    global qid_to_pids_gt, qid_to_pids_pred
    qid_to_pids_gt = Y_df.groupby('query')['passage'].unique()
    qid_to_pids_pred = Y_pred_df.groupby('query')['passage'].unique()
    
    print(qid_to_pids_gt.shape, qid_to_pids_pred.shape)
    
    qids = list(qid_to_pids_gt.keys())
    assert(qids == sorted(qids))
    
    _mine_one = partial(mine_one, depth=args.depth, n_sample=args.n_sample)
    
    with Pool(processes=args.mp_threads) as p:
        out = list(tqdm(p.imap(_mine_one, qids, chunksize=args.mp_chunk_size), total=len(qids)))
    
    out = np.concatenate(out, axis=0)
    Y_col, Y_row, Y_data = out[:, 0], out[:, 1], out[:, 2]
    
    # convert x dims
    qid_to_i_trn = {qid: i for i, qid in enumerate(qids)}
    Y_col = np.array(list(map(lambda x: qid_to_i_trn[x], Y_col)))
    
    print(out.shape, Y_col.min(), Y_col.max(), Y_row.min(), Y_row.max())
    
    Y = sparse.csr_matrix((Y_data, (Y_col, Y_row)), shape=(nx_trn, nz))
    print(Y.shape, np.unique(Y.data, return_counts=True))
    
    sparse.save_npz(args.save_Y_path, Y, compressed=False)
    
if __name__ == '__main__':
    main()
