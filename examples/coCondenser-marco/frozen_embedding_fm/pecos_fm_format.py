import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm, trange
import itertools
from scipy import sparse
import time
import random

def pair_collate_fn(xzs):
    xs = torch.stack([x_ for x_, zs_ in xzs], dim=0)
    zs = torch.cat([zs_ for x_, zs_ in xzs], dim=0)
    return xs, zs

class PecosFMDataset(Dataset):
    
    def __init__(self, X, Z, Y, n_train_zs, normalize_emb=False):
        
        self.X = torch.as_tensor(X) # avoids copying X compared to from_numpy
        self.Z = torch.as_tensor(Z)
        
        self.Y = Y.copy() # [-1, 1]
        if set(self.Y.data) != set([-1, 1]):
            raise ValueError(f"Expect Y to contain [-1, 1] but got {set(self.Y.data)}.")

        self.Y_nz_rows = np.array([i for i in range(self.Y.shape[0]) 
                                   if self.Y.getrow(i).nnz > 0])
            
        self.n_train_zs = n_train_zs
        self.normalize_emb = normalize_emb
    
    def __len__(self):
        return len(self.Y_nz_rows)
    
    def __getitem__(self, idx):
        
        r = self.Y_nz_rows[idx]
        x = self.X[r]
        yi = self.Y.getrow(r)
        
        # filter out positive and negative columns
        c_poss = yi.indices[yi.data == 1].tolist()
        c_negs = yi.indices[yi.data == -1].tolist()
        
        # random sample one pos
        if len(c_poss) == 0:
            c_poss = [i for i in range(self.X.shape[0]) if i not in c_negs]
        c_pos = random.choice(c_poss)
        
        # sample some negatives
        n_negs = self.n_train_zs - 1
        if n_negs == 0:
            c_negs = []
        elif len(c_negs) == 0:
            candidates = [i for i in range(self.Z.shape[0]) if i not in c_poss]
            c_negs = random.choices(candidates, k=n_negs)
        elif len(c_negs) < n_negs:
            c_negs = random.choices(c_negs, k=n_negs)
        else:
            random.shuffle(c_negs)
            c_negs = c_negs[:n_negs]

        # combine positive and negatives
        cs = np.array([c_pos] + c_negs)

        zs = self.Z[cs]
        
        if self.normalize_emb:
            x /= torch.norm(x)
            zs /= torch.norm(zs, dim=1)
        
        return x, zs

class PecosBCEDataset(Dataset):
    
    def __init__(self, X, Z, Y, normalize_emb=False, shuffle_row=True):
        
        self.X = torch.as_tensor(X) # avoids copying X compared to from_numpy
        self.Z = torch.as_tensor(Z)
        self.shuffle_row = shuffle_row
        
        if self.shuffle_row:
            self.Y = Y.copy() 
            self.Y.data = (self.Y.data > 0).astype(float) # [-1, 1] to [0, 1]

            self.row_ordering = np.arange(self.Y.shape[0])
            self.cum_col = np.cumsum([self.Y.getrow(ri).nnz for ri in self.row_ordering])
        else:
            self.Y = Y.tocoo()
            self.Y.data = (self.Y.data > 0).astype(float) # [-1, 1] to [0, 1]
        
        self.normalize_emb = normalize_emb
    
    def shuffle(self):
        assert(self.shuffle_row)
        
        np.random.shuffle(self.row_ordering)
        self.cum_col = np.cumsum([self.Y.getrow(ri).nnz for ri in self.row_ordering])
    
    def __len__(self):
        return self.Y.nnz
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.shuffle_row:
            outer_r = np.argmax(self.cum_col > idx)
            outer_c = idx if outer_r == 0 else idx - self.cum_col[outer_r - 1]

            r = self.row_ordering[outer_r]
            yi = self.Y.getrow(r)
            c, y = yi.indices[outer_c], yi.data[outer_c]
        else:
            r, c, y = self.Y.row[idx], self.Y.col[idx], self.Y.data[idx]
        
        x = self.X[r]
        z = self.Z[c]
        
        if self.normalize_emb:
            x /= torch.norm(x)
            z /= torch.norm(z)
        
        y = torch.as_tensor(y)
        
        return x, z, y
