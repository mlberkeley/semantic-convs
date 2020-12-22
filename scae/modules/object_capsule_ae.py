import numpy as np
import torch.nn as nn
import torch
import modules.attention as _attention
import modules.capsule as _capsule
from attrdict import AttrDict
from util import math as math
import torch.nn.functional as F
import torch.distributions as D
import collections


class SetTransformer():

    """Encoder."""

    def __init__(self):
        pass


class ImageCapsule(nn.Module):

    """Decoder."""

    def __init__(self, n_caps, n_caps_dims, n_votes, **capsule_kwargs):
        super(ImageCapsule, self).__init__()
        self._n_caps = n_caps
        self._n_caps_dims = n_caps_dims
        self._n_votes = n_votes
        self._capsule_kwargs = capsule_kwargs
        self.build()
    
    def build(self):
        self.capsule = _capsule.CapsuleLayer(self._n_caps, self._n_caps_dims,
                                    self._n_votes, **self._capsule_kwargs)
    
    def forward(self, h, x, presence=None):
        batch_size = int(x.shape[0])
        self.vote_shape = [batch_size, self._n_caps, self._n_votes, 6]
        res = self.capsule(h)
        res.vote = torch.reshape(res.vote[..., :-1, :], self.vote_shape)
        votes, scale, vote_presence_prob = res.vote, res.scale, res.vote_presence
        self.likelihood = _capsule.CapsuleLikelihood(votes, scale,
                                                     vote_presence_prob)
        ll_res = self.likelihood(x, presence)
        res.update(ll_res._asdict())
        caps_presence_prob = torch.max(torch.reshape(vote_presence_prob,
                   [batch_size, self._n_caps, self._n_votes]), dim=2)
        res.caps_presence_prob = caps_presence_prob
        return res
        
        
        
        
        
    

