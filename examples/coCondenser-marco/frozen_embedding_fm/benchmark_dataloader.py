import argparse
import os
import sys
import logging
import numpy as np
from scipy import sparse
from tqdm.auto import tqdm, trange
import json
from pathlib import Path
from collections import namedtuple
import time

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy

from torchfm.dataset.pecos_fm_format import PecosFMDataset
from torchfm.model.efm import EmbeddingFactorizationMachineModel

from train import load_data

def main():
    data_dir = '../data/MS-MARCO/original/'
    bsize = 2048
    num_workers = 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_trn, X_tst, Y_trn, Y_tst, Z = load_data(data_dir)
    
    start_t = time.time()
    
    dset_trn = PecosFMDataset(X_trn['embs'], Z['embs'], Y_trn)
    dl_trn = DataLoader(dset_trn, batch_size=bsize, shuffle=False, num_workers=num_workers)
    
    print(time.time() - start_t)
    start_t = time.time()
    
    print(dset_trn[0][0][:10])
    dset_trn.shuffle()
    print(dset_trn[0][0][:10])
    
    print(time.time() - start_t)
    start_t = time.time()
    
    
    
    for i, (xs, zs, ys) in enumerate(tqdm(dl_trn)):
        xs, zs, ys = xs.to(device), zs.to(device), ys.to(device)
        if i == 0:
            print(time.time() - start_t)
            print(xs[0, :10])
        break

if __name__ == '__main__':
    main()