import json
from collections import namedtuple
import torch
import os
import math

class SymmetricLinear(torch.nn.Module):
    def __init__(self, in_out_dim, identity_factor=1):
        super().__init__()
        stdv = 1. / math.sqrt(in_out_dim)
        self.Q = torch.nn.parameter.Parameter(torch.empty((in_out_dim, in_out_dim)))
        self.diag = torch.nn.parameter.Parameter(torch.empty(in_out_dim))
        torch.nn.init.uniform_(self.Q, -stdv, stdv)
        torch.nn.init.uniform_(self.diag, -stdv, stdv)
        
        with torch.no_grad():
            self.Q += torch.eye(in_out_dim) * identity_factor
            self.diag += torch.ones(in_out_dim) * identity_factor
            
        
    def forward(self, x):
        x = torch.matmul(x, self.Q)
        x = torch.matmul(x, torch.diag(self.diag))
        x = torch.matmul(x, self.Q.T)
        return x

class EmbeddingFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine with dense embedding as input.
    """

    def __init__(self, k, x_emb_dim, z_emb_dim, use_bias=False, ip=False, identity_factor=1.0, 
                 share_bias=False, use_fm=False, tanh=False, q_mode='asymmetric', normalize=False, **kwargs):
        super().__init__()
        
        assert((use_bias or use_fm) ^ ip)
        
        self.use_bias = use_bias
        self.ip = ip
        self.use_fm = use_fm
        self.tanh = tanh
        self.normalize = normalize
        
        if self.use_bias:
            self.x_bias = torch.nn.Linear(x_emb_dim, 1, bias=False)
            self.z_bias = sefl.x_bias if self.share_bias else torch.nn.Linear(z_emb_dim, 1, bias=False)
        
        if q_mode == 'asymmetric':
            self.x_dim_reduc = torch.nn.Linear(x_emb_dim, k, bias=False)
            self.z_dim_reduc = torch.nn.Linear(z_emb_dim, k, bias=False)
            
            # init as identity + some noise
            with torch.no_grad():
                im = torch.zeros_like(self.x_dim_reduc.weight)
                im.fill_diagonal_(1)
                self.x_dim_reduc.weight += im * identity_factor

                im = torch.zeros_like(self.z_dim_reduc.weight)
                im.fill_diagonal_(1)
                self.z_dim_reduc.weight += im * identity_factor
                
        elif q_mode == 'symmetric':
            assert(k == x_emb_dim)
            assert(k == z_emb_dim)
            self.x_dim_reduc = SymmetricLinear(k, identity_factor)
            self.z_dim_reduc = torch.nn.Identity()
        
        elif q_mode == 'psd':
            assert(k == x_emb_dim)
            assert(k == z_emb_dim)
            self.x_dim_reduc = torch.nn.Linear(x_emb_dim, k, bias=False)
            self.z_dim_reduc = self.x_dim_reduc
            
            # init as identity + some noise
            with torch.no_grad():
                im = torch.zeros_like(self.x_dim_reduc.weight)
                im.fill_diagonal_(1)
                self.x_dim_reduc.weight += im * identity_factor
                
        elif q_mode == 'q_only':
            self.x_dim_reduc = torch.nn.Linear(x_emb_dim, k, bias=False)
            with torch.no_grad():
                im = torch.zeros_like(self.x_dim_reduc.weight)
                im.fill_diagonal_(1)
                self.x_dim_reduc.weight += im * identity_factor
            self.z_dim_reduc = torch.nn.Identity()
        
        elif q_mode == 'p_only':
            self.z_dim_reduc = torch.nn.Linear(z_emb_dim, k, bias=False)
            with torch.no_grad():
                im = torch.zeros_like(self.z_dim_reduc.weight)
                im.fill_diagonal_(1)
                self.z_dim_reduc.weight += im * identity_factor
            self.x_dim_reduc = torch.nn.Identity()
        
        elif q_mode == 'identity':
            assert(k == x_emb_dim)
            assert(k == z_emb_dim)
            self.x_dim_reduc = torch.nn.Identity()
            self.z_dim_reduc = torch.nn.Identity()
            
        else:
            raise NotImplementedError
    
    @staticmethod
    def load(model_dir, ckpt_path=None):
        
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            train_args = json.load(f)
        
        model = EmbeddingFactorizationMachineModel(**train_args)
        if ckpt_path is None:
            ckpt_path = os.path.join(model_dir, 'fm.pth')
        model.load_state_dict(torch.load(ckpt_path))
        
        return model
        
    def forward(self, x, z):
        """
        :param x: Long tensor of size ``(batch_size, emb_dim)``
        """
        ex = self.encode_x(x)
        ez = self.encode_z(z)
        
        out = torch.matmul(ex, ez.T)
        # out = (ex * ez).sum(1)
        return out

    def encode_x(self, x):
        ex = self.x_dim_reduc(x) # [bsize, hdim] -> [bsize, k]
        
        if self.ip:
            return ex
        
        # self.x_dim_reduc.weights\.shape = [k, d]
        x_bias = 0
        
        if self.use_bias:
            x_bias += self.x_bias(x).squeeze(1)
        
        if self.use_fm:
            q = (ex**2).sum(1) - ((x**2) * (self.x_dim_reduc.weight**2).sum(0, keepdim=True)).sum(1)
            x_bias += 0.5 * q
        
        ex = torch.cat([ex, x_bias.unsqueeze(1), torch.ones_like(x_bias).unsqueeze(1)], dim=1)
        
        if self.tanh:
            ex = torch.tanh(ex)
            
        if self.normalize:
            ex /= torch.norm(ex, dim=1, keepdim=True)
        
        return ex
        
    def encode_z(self, z):
        ez = self.z_dim_reduc(z)
        
        if self.ip:
            return ez
        
        z_bias = 0
        
        if self.use_bias:
            z_bias += self.z_bias(z).squeeze(1)
        
        if self.use_fm:
            q = (ez**2).sum(1) - ((z**2) * (self.z_dim_reduc.weight**2).sum(0, keepdim=True)).sum(1)
            z_bias += 0.5 * q
        
        ez = torch.cat([ez, torch.ones_like(z_bias).unsqueeze(1), z_bias.unsqueeze(1)], dim=1)

        if self.tanh:
            ez = torch.tanh(ez)
            
        if self.normalize:
            ez /= torch.norm(ez, dim=1, keepdim=True)
        
        return ez
