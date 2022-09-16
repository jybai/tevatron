import argparse
import os
import numpy as np
from scipy import sparse
import pandas as pd
from pecos.utils.smat_util import Metrics

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("qrels_path", help="Path relavant passages to query ground truth path.", type=str)
    parser.add_argument("pred_rank_path", help="Path to predicted rank", type=str)
    parser.add_argument('--topk', type=int, default=10, help='Eval topk results') 
    
    return parser.parse_args()

def main():
    
    args = parse_args()
    
    # load gt
    Y_df = pd.read_csv(args.qrels_path, header=None, delimiter='\t')
    Y_row = Y_df.iloc[:, 0].values.astype(int)
    if 'train' in args.qrels_path:
        Y_col = Y_df.iloc[:, 2].values.astype(int)
    elif 'dev' in args.qrels_path:
        Y_col = Y_df.iloc[:, 1].values.astype(int)
    else:
        raise NotImplementedError
    
    '''
    Y_row, Y_col = [], []
    
    with open(args.qrels_path, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            vals = line.split('\t')
            Y_row.append(int(vals[0]))
            if 'train' in args.qrels_path:
                Y_col.append(int(vals[2]))
            elif 'dev' in args.qrels_path:
                Y_col.append(int(vals[1]))
            else:
                raise NotImplementedError
    '''
    
    # load rank.tsv
    Y_pred_df = pd.read_csv(args.pred_rank_path, header=None, delimiter='\t')
    Y_pred_row = Y_pred_df.iloc[:, 0].values.astype(int)
    Y_pred_col = Y_pred_df.iloc[:, 1].values.astype(int)
    Y_pred_data = Y_pred_df.iloc[:, 2].values.astype(float)
    
    '''
    Y_pred_row, Y_pred_col, Y_pred_data = [], [], []
    with open(args.pred_rank_path, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            vals = line.split('\t')
            Y_pred_row.append(int(vals[0]))
            Y_pred_col.append(int(vals[1]))
            Y_pred_data.append(float(vals[2]))
    '''
    
    max_row = max(np.max(Y_row), np.max(Y_pred_row)) + 1
    max_col = max(np.max(Y_col), np.max(Y_pred_col)) + 1
    
    # construct Y_gt_csr
    Y = sparse.csr_matrix((np.ones(len(Y_row)), (Y_row, Y_col)), shape=(max_row, max_col))
    # print(Y.shape, Y.nnz)
    
    # construct Y_pred_csr
    Y_pred = sparse.csr_matrix((Y_pred_data, (Y_pred_row, Y_pred_col)), shape=(max_row, max_col))
    # print(Y_pred.shape, Y_pred.nnz)
    
    # eval metric
    print(Metrics.generate(Y, Y_pred, topk=args.topk))
    
    
if __name__ == '__main__':
    main()