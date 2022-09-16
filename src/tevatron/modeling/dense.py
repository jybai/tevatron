import math
import torch
import torch.nn as nn
from torch import Tensor
import logging
from .encoder import EncoderPooler, EncoderModel

logger = logging.getLogger(__name__)

class SymmetricLinear(nn.Module):
    def __init__(self, in_out_dim, proj=True):
        super().__init__()
        self.proj = proj
        stdv = 1. / math.sqrt(in_out_dim)
        if self.proj:
            self.Q = nn.parameter.Parameter(torch.empty((in_out_dim, in_out_dim)))
            nn.init.uniform_(self.Q, -stdv, stdv)
        self.diag = nn.parameter.Parameter(torch.empty(in_out_dim))
        nn.init.uniform_(self.diag, -stdv, stdv)
        
    def forward(self, x):
        if self.proj:
            x = torch.matmul(x, self.Q)
        x = torch.matmul(x, torch.diag(self.diag))
        if self.proj:
            x = torch.matmul(x, self.Q.T)
        return x

class DensePooler(EncoderPooler):
    def __init__(self, input_dim: int = 768, output_dim: int = 768, tied=True, symmetric=False):
        super(DensePooler, self).__init__()
        
        if symmetric:
            assert(input_dim == output_dim)
            self.linear_q = SymmetricLinear(input_dim)
            self.linear_p = nn.Identity()
        else:
            self.linear_q = nn.Linear(input_dim, output_dim, bias=False)
            if tied:
                self.linear_p = self.linear_q
            else:
                self.linear_p = nn.Linear(input_dim, output_dim, bias=False)
        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 'tied': tied, 
                        'symmetric': symmetric}

    def forward(self, q: Tensor = None, p: Tensor = None, **kwargs):
        if q is not None:
            pool_q = self.linear_q(q[:, 0])
            return pool_q
        elif p is not None:
            pool_p = self.linear_p(p[:, 0])
            return pool_p
        else:
            raise ValueError
            
    def init_weight_identity(self, identity_bias_magnitude):
        with torch.no_grad():
            if self._config['symmetric']:
                self.linear_q.diag += torch.ones_like(self.linear_q.diag) * identity_bias_magnitude
                im = torch.zeros_like(self.linear_q.Q)
                im.fill_diagonal_(1)
                self.linear_q.Q += im * identity_bias_magnitude
            else:
                im = torch.zeros_like(self.linear_q.weight)
                im.fill_diagonal_(1)
                self.linear_q.weight += im * identity_bias_magnitude

                if not self._config['tied']:
                    im = torch.zeros_like(self.linear_p.weight)
                    im.fill_diagonal_(1)
                    self.linear_p.weight += im * identity_bias_magnitude
                
class DenseModel(EncoderModel):
    def encode_passage(self, psg):
        if psg is None:
            return None
        psg_out = self.lm_p(**psg, return_dict=True)
        p_hidden = psg_out.last_hidden_state
        if self.pooler is not None:
            p_reps = self.pooler(p=p_hidden)  # D * d
        else:
            p_reps = p_hidden[:, 0]
        return p_reps

    def encode_query(self, qry):
        if qry is None:
            return None
        qry_out = self.lm_q(**qry, return_dict=True)
        q_hidden = qry_out.last_hidden_state
        if self.pooler is not None:
            q_reps = self.pooler(q=q_hidden)
        else:
            q_reps = q_hidden[:, 0]
        return q_reps

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    @staticmethod
    def load_pooler(model_weights_file, **config):
        pooler = DensePooler(**config)
        pooler.load(model_weights_file)
        return pooler

    @staticmethod
    def build_pooler(model_args):
        pooler = DensePooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_pooler,
            symmetric=model_args.pooler_symmetric,
        )
            
        pooler.load(model_args.model_name_or_path)
        return pooler
