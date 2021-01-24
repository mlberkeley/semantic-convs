"""
attention.py
~~~~
Code taken and modified from official implementation of SetTransformer:
https://github.com/juho-lee/set_transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SetTransformer(nn.Module):

    def __init__(self,
                 n_input_dims,  # Last dimension of input
                 n_layers=5,  # number of attention blocks
                 n_heads=4,
                 n_dims=128,  # hidden layer dimension
                 n_output_dims=32,
                 n_outputs=3,
                 layer_norm=False,
                 n_inducing_points=32):

        super(SetTransformer, self).__init__()

        enc_modules = [ISAB(n_dims, n_dims, n_heads, n_inducing_points,
                            ln=layer_norm) for i in range(n_layers-1)]
        self.enc = nn.Sequential(ISAB(n_input_dims, n_dims, n_heads,
                                      n_inducing_points, ln=layer_norm),
                                 *enc_modules)

        dec_modules = [SAB(n_dims, n_dims, n_heads, ln=layer_norm)
                       for i in range(n_layers)]
        self.dec = nn.Sequential(
            PMA(n_dims, n_heads, n_outputs, ln=layer_norm),
            SAB(n_dims, n_dims, n_heads, ln=layer_norm),
            *dec_modules,
            nn.Linear(n_dims, n_output_dims))

    def forward(self, X):
        print("INITIAL X", X.shape)
        return self.dec(self.enc(X))


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, 40*dim_V)
        self.fc_v = nn.Linear(dim_K, 40*dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        
        print("Q shape:", Q.shape, ", Layer:", self.fc_q.in_features,
              self.fc_q.out_features)
        
        Q = self.fc_q(Q)
        K = torch.flatten(K, start_dim=1)
        
        print("K shape:", K.shape, ", Layer:", self.fc_k.in_features,
              self.fc_k.out_features)
        print("V shape:", K.shape, ", Layer:", self.fc_v.in_features,
              self.fc_v.out_features)
        
        K, V = self.fc_k(K), self.fc_v(K)
        
        # make more robust as it is really (Batch, n_templates, n_dims)
        K = torch.reshape(K, (128, 40, 128))
        V = torch.reshape(K, (128, 40, 128))
        
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, 40*dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(40*dim_in, 32*dim_out, 32*dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(torch.flatten(X, start_dim=1), H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
