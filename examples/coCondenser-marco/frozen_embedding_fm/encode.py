import argparse
import os
import numpy as np
from scipy import sparse
from tqdm.auto import tqdm, trange
from collections import namedtuple
import json
import pickle

import torch
from torch.utils.data import DataLoader, TensorDataset

from torchfm.dataset.pecos_fm_format import PecosFMDataset
from torchfm.model.efm import EmbeddingFactorizationMachineModel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("data_dir", help="Directory to data.", type=str)
    parser.add_argument("model_dir", help="Directory to load saved model.", type=str)
    parser.add_argument("emb_dir", help="Directory to save embeddings.", type=str)
    
    parser.add_argument('--bsize', type=int, default=2048, help='Batch size. Does not affect result. Use max bsize that fits in GPU for fastest encoding.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for loading data.')
    parser.add_argument('--pickle', action='store_true')
    parser.add_argument('--ip', action='store_true')
    
    return parser.parse_args()

def main():
    
    args = parse_args()
    with open(os.path.join(args.model_dir, 'config.json'), 'r') as f:
        train_args = json.load(f)
        train_args = namedtuple("train_args", train_args.keys())(*train_args.values())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with np.load(os.path.join(args.data_dir, 'X.trn.npz')) as f:
        X_trn = f['embs']
        X_trn_indices = f['indices']
    with np.load(os.path.join(args.data_dir, 'X.tst.npz')) as f:
        X_tst = f['embs']
        X_tst_indices = f['indices']
    with np.load(os.path.join(args.data_dir, 'Z.npz')) as f:
        Z = f['embs']
        Z_indices = f['indices']
    
    X_trn_dset = TensorDataset(torch.as_tensor(X_trn))
    X_tst_dset = TensorDataset(torch.as_tensor(X_tst))
    Z_dset = TensorDataset(torch.as_tensor(Z))
    X_trn_dl = DataLoader(X_trn_dset, batch_size=args.bsize, shuffle=False, num_workers=args.num_workers)
    X_tst_dl = DataLoader(X_tst_dset, batch_size=args.bsize, shuffle=False, num_workers=args.num_workers)
    Z_dl = DataLoader(Z_dset, batch_size=args.bsize, shuffle=False, num_workers=args.num_workers)
    
    fm = EmbeddingFactorizationMachineModel(train_args.k, X_trn.shape[1], Z.shape[1], 
                                            use_bias=train_args.use_bias, ip=train_args.ip,
                                            use_fm=args.use_fm, tanh=args.tanh)
    
    fm.load_state_dict(torch.load(os.path.join(args.model_dir, 'fm.pth')))
    fm.eval().to(device)
    
    exs_trn, exs_tst, ezs = [], [], []
    fm.eval()
    with torch.no_grad():
        for xs in tqdm(X_trn_dl):
            xs = xs[0].to(device)
            exs_ = fm.encode_x(xs).detach().cpu().numpy()
            exs_trn.append(exs_)
        for xs in tqdm(X_tst_dl):
            xs = xs[0].to(device)
            exs_ = fm.encode_x(xs).detach().cpu().numpy()
            exs_tst.append(exs_)
        for zs in tqdm(Z_dl):
            zs = zs[0].to(device)
            ezs_ = fm.encode_z(zs).detach().cpu().numpy()
            ezs.append(ezs_)
    exs_trn = np.concatenate(exs_trn, axis=0)
    exs_tst = np.concatenate(exs_tst, axis=0)
    ezs = np.concatenate(ezs, axis=0)
    
    if args.ip:
        exs_trn = exs_trn[:, :-2]
        exs_tst = exs_tst[:, :-2]
        ezs = ezs[:, :-2]
    
    print(exs_trn.shape, exs_tst.shape, ezs.shape)
    '''
    for i in range(10):
        print(np.dot(exs_trn[i], ezs[i]), 
              fm(X_trn_dset[i][0].to(device).unsqueeze(0), Z_dset[i][0].to(device).unsqueeze(0)).item())
    '''
    
    os.makedirs(args.emb_dir, exist_ok=True)
    
    if args.pickle:
        with open(os.path.join(args.emb_dir, 'X.trn.pkl'), "wb") as f:
            pickle.dump([exs_trn, X_trn_indices.tolist()], f)
        with open(os.path.join(args.emb_dir, 'X.tst.pkl'), "wb") as f:
            pickle.dump([exs_tst, X_tst_indices.tolist()], f)
        with open(os.path.join(args.emb_dir, 'Z.pkl'), "wb") as f:
            pickle.dump([ezs, Z_indices.tolist()], f)
    else:
        np.savez(os.path.join(args.emb_dir, 'X.trn.npz'), embs=exs_trn, indices=X_trn_indices)
        np.savez(os.path.join(args.emb_dir, 'X.tst.npz'), embs=exs_tst, indices=X_tst_indices)
        np.savez(os.path.join(args.emb_dir, 'Z.npz'), embs=ezs, indices=Z_indices)
    
if __name__ == '__main__':
    main()