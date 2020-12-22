import numpy as np
import torch.nn as nn
import torch
from attrdict import AttrDict
from util import math as math
import torch.nn.functional as F
import torch.distributions as D
import collections


class CapsuleLayer(nn.Module):
    _n_transform_params = 6
    def __init__(self, n_caps, n_caps_dims, n_votes, n_caps_params=None,
               n_hiddens=128, learn_vote_scale=False, deformations=True,
               noise_type=None, noise_scale=0., similarity_transform=True,
               caps_dropout_rate=0.0):
      super(CapsuleLayer, self).__init__()
      self._n_caps = n_caps
      self._n_caps_dims = n_caps_dims
      self._n_caps_params = n_caps_params

      self._n_votes = n_votes
      self._n_hiddens = n_hiddens
      self._learn_vote_scale = learn_vote_scale
      self._deformations = deformations
      self._noise_type = noise_type
      self._noise_scale = noise_scale

      self._similarity_transform = similarity_transform
      self._caps_dropout_rate = caps_dropout_rate

      assert n_caps_dims == 2, ('This is the only value implemented now due to '
                              'the restriction of similarity transform.')
      
      self.output_shapes = (
            [self._n_votes, self._n_transform_params],  # CPR_dynamic
            [1, self._n_transform_params],  # CCR
            [1],  # per-capsule presence
            [self._n_votes],  # per-vote-presence
            [self._n_votes],  # per-vote scale
            )
      self.splits = [np.prod(i).astype(np.int32) for i in self.output_shapes]
      self.n_outputs = sum(self.splits)
      
      self.build()
      
    def build(self):
        self.mlp = nn.ModuleList()
        shape_list = [16, self._n_hiddens, self._n_caps_params]
        for i in range(1, len(shape_list)):
            self.mlp.append(nn.Linear(shape_list[i-1], shape_list[i]))
            self.mlp.append(nn.ReLU())
        
        self.caps_mlp = nn.ModuleList()
        shape_list = [32, self._n_hiddens, self.n_outputs]
        for i in range(1, len(shape_list)):
            self.caps_mlp.append(nn.Linear(shape_list[i-1],
                                           shape_list[i], bias = False))
            self.caps_mlp.append(nn.ReLU())
        
        
    def forward(self, x, parent_transform=None, parent_presence=None):
        batch_size = x.shape[0]
        batch_shape = [batch_size, self._n_caps]
        if self._n_caps_params is not None:
            raw_caps_params = self.mlp(x)
            caps_params = torch.reshape(raw_caps_params,
                                   batch_shape + [self._n_caps_params])
        else:
            assert x.shape[:2].as_list() == batch_shape
            caps_params = x
        
        if self._caps_dropout_rate == 0.0:
            caps_exist = torch.ones(batch_shape + [1], dtype=torch.float32)
        else:
            pmf = D.bernoulli.Bernoulli(1. - self._caps_dropout_rate,
                                        dtype=torch.float32)
            caps_exist = pmf.sample(batch_shape + [1])
        
        caps_params = torch.cat([caps_params, caps_exist], -1)
        all_params = self.caps_mlp(caps_params)
        all_params = torch.split(all_params, self.splits, -1)
        res = [torch.reshape(i, batch_shape + s)
           for (i, s) in zip(all_params, self.output_shapes)]
        cpr_dynamic = res[0]
        
        # res = [snt.AddBias()(i) for i in res[1:]]
        ccr, pres_logit_per_caps, pres_logit_per_vote, scale_per_vote = res
        
        if self._caps_dropout_rate != 0.0:
            pres_logit_per_caps += math.safe_log(caps_exist)
        
        cpr_static = torch.Tensor(size=[1, self._n_caps, self._n_votes,
                                        self._n_transform_params],
                                  requires_grad=True)
        
        pres_logit_per_caps = math.add_noise(pres_logit_per_caps)
        pres_logit_per_vote = math.add_noise(pres_logit_per_vote)       
        
        if parent_transform is None:
            ccr = math.geometric_transform(ccr)
        else:
            ccr = parent_transform
        if not self._deformations:
            cpr_dynamic = torch.zeros_like(cpr_dynamic)
        
        cpr = math.geometric_transform(cpr_dynamic + cpr_static)
        ccr_per_vote = ccr.tile((2, self._n_votes))
        votes = torch.matmul(ccr_per_vote, cpr)
        
        if parent_presence is not None:
            pres_per_caps = parent_presence
        else:
            pres_per_caps = torch.sigmoid(pres_logit_per_caps)
        
        pres_per_vote = pres_per_caps * torch.sigmoid(pres_logit_per_vote)

        if self._learn_vote_scale:
            scale_per_vote = F.softplus(scale_per_vote + .5) + 1e-2
        else:
            scale_per_vote = torch.zeros_like(scale_per_vote) + 1.

        return AttrDict(
            vote=votes,
            scale=scale_per_vote,
            vote_presence=pres_per_vote,
            pres_logit_per_caps=pres_logit_per_caps,
            pres_logit_per_vote=pres_logit_per_vote,
            dynamic_weights_l2=0.5*torch.square(cpr_dynamic) / batch_size,
            raw_caps_params=raw_caps_params,
            raw_caps_features=x,
        )



class OrderInvariantCapsuleLikelihood(nn.Module):
    OutputTuple = collections.namedtuple('CapsuleLikelihoodTuple',  # pylint:disable=invalid-name
                                       ('log_prob vote_presence winner '
                                        'winner_pres is_from_capsule '
                                        'mixing_logits mixing_log_prob '
                                        'soft_winner soft_winner_pres '
                                        'posterior_mixing_probs'))
    def __init__(self, n_votes, votes, scales, vote_presence_prob,
                 pdf='normal'):
        super(OrderInvariantCapsuleLikelihood, self).__init__()
        self._n_votes = n_votes
        self._votes = votes
        self._scales = scales
        self._vote_presence_prob = vote_presence_prob
        self._pdf = pdf
        self.build()
    
    def build(self):
        self.expanded_votes = self._votes.unsqueeze(dim=1)
        self.expanded_scale = self._scales.unsqueeze(dim=1).unsqueeze(dim=-1)
        self.vote_component_pdf = self._get_pdf(self.expanded_votes,
                                                self.expanded_scale)
        self.mixing_logits = math.safe_log(self._vote_presence_prob)
    
    def forward(self, x, presence=None):
        self.batch_size, self.n_input_points = x.shape[:2].as_list()
        expanded_x = x.unsqueeze(dim=2)
        vote_log_prob_per_dim = self.vote_component_pdf.log_prob(expanded_x)
        vote_log_prob = torch.sum(vote_log_prob_per_dim, dim=-1)
        dummy_vote_log_prob = torch.zeros(
            [self.batch_size, self.n_input_points, 1])
        dummy_vote_log_prob -= 2. * torch.log(10.)
        vote_log_prob = torch.cat([vote_log_prob, dummy_vote_log_prob], 2)
        mixing_logits = math.safe_log(self._vote_presence_prob)
        dummy_logit = torch.zeros([self.batch_size, 1]) -2. * torch.log(10.)
        mixing_logits = torch.cat([mixing_logits, dummy_logit], 1)

        mixing_log_prob = mixing_logits - torch.logsumexp(mixing_logits,dim=1,
                                                          keepdims=True)

        expanded_mixing_logits = mixing_log_prob.unsqueeze(dim=1)
        mixture_log_prob_per_component= torch.logsumexp(
            expanded_mixing_logits + vote_log_prob, dim=2)

        if presence is not None:
            presence = presence.type(torch.FloatTensor)
            mixture_log_prob_per_component *= presence

        mixture_log_prob_per_example = torch.sum(mixture_log_prob_per_component,
                                             dim=1)

        mixture_log_prob_per_batch = torch.sum(mixture_log_prob_per_example)

        # [B, n_points, n_votes]
        posterior_mixing_logits_per_point = (expanded_mixing_logits +
                                             vote_log_prob)
        # [B, n_points]
        winning_vote_idx = torch.argmax(
            posterior_mixing_logits_per_point[:, :, :-1], dim=2)

        batch_idx = torch.range(self.batch_size,
                                dtype=torch.int64).unsqueeze(dim=-1)
        batch_idx = batch_idx.tile((1, winning_vote_idx.shape[-1]))
        idx = torch.stack([batch_idx, winning_vote_idx], dim=-1)
        winning_vote = torch.index_select(self._votes, idx)
        winning_pres = torch.index_select(self._vote_presence_prob, idx)
        vote_presence = torch.gt(mixing_logits[:, :-1],
                               mixing_logits[:, -1:])

        # the first four votes belong to the square
        is_from_capsule = winning_vote_idx // self._n_votes

        posterior_mixing_probs = F.softmax(
            posterior_mixing_logits_per_point, dim=-1)[..., :-1]

        assert winning_vote.shape == x.shape

        return self.OutputTuple(
            log_prob=mixture_log_prob_per_batch,
            vote_presence=vote_presence.type(torch.FloatTensor),
            winner=winning_vote,
            winner_pres=winning_pres,
            is_from_capsule=is_from_capsule,
            mixing_logits=mixing_logits,
            mixing_log_prob=mixing_log_prob,
            # TODO(adamrk): this is broken
            soft_winner=torch.zeros_like(winning_vote),
            soft_winner_pres=torch.zeros_like(winning_pres),
            posterior_mixing_probs=posterior_mixing_probs,
        )
    
    def _get_pdf(self, votes, scale):
        if self._pdf == 'normal':
          pdf = torch.Normal(votes, scale)
        else:
          raise ValueError('Distribution "{}" not '
                           'implemented!'.format(self._pdf))
        return pdf



class CapsuleLikelihood(OrderInvariantCapsuleLikelihood):
    def __init__(self, votes, scales, vote_presence_prob, pdf='normal'):
        super(CapsuleLikelihood, self).__init__(1, votes, scales,
                                            vote_presence_prob, pdf)
        self._n_caps = int(self._votes.shape[1])
        self.vote_component_pdf = self._get_pdf(self._votes,
                                       self._scales.unsqueeze(dim=-1))
    def forward(self, x, presence=None):
        batch_size, n_input_points = x.shape[:2].as_list()
        expanded_x = x.unsqueeze(dim=1)
        vote_log_prob_per_dim = self.vote_component_pdf.log_prob(expanded_x)
        vote_log_prob = torch.sum(vote_log_prob_per_dim, dim=-1)
        dummy_vote_log_prob = torch.zeros([batch_size, 1, n_input_points])
        dummy_vote_log_prob -= 2. * torch.log(10.)
        vote_log_prob = torch.cat([vote_log_prob, dummy_vote_log_prob], dim=1)
        mixing_logits = math.safe_log(self._vote_presence_prob)
        dummy_logit = torch.zeros([batch_size, 1, 1]) -2. * torch.log(10.)
        dummy_logit = dummy_logit.tile((2, self.n_input_points))
        mixing_logits = torch.cat([mixing_logits, dummy_logit], dim=1)
        mixing_log_prob = mixing_logits - torch.logsumexp(mixing_logits, dim=1,
                                                          keepdims=True)
        mixture_log_prob_per_point = torch.logsumexp(
            mixing_logits + vote_log_prob, dim=1)
        if presence is not None:
            presence = presence.type(torch.FloatTensor)
            mixture_log_prob_per_point *= presence
        mixture_log_prob_per_example = torch.sum(
            mixture_log_prob_per_point, dim=1)
        mixture_log_prob_per_batch = torch.mean(
            mixture_log_prob_per_example)
        posterior_mixing_logits_per_point = mixing_logits + vote_log_prob
        winning_vote_idx = torch.argmax(
            posterior_mixing_logits_per_point[:, :-1], dim=1)
        batch_idx = torch.range(batch_size,
                                dtype=torch.int64).unsqueeze(dim=1)
        batch_idx = batch_idx.tile((1, self.n_input_points))
        point_idx = torch.range(self.n_input_points,
                                dtype=torch.int64).unsqueeze(dim=0)
        point_idx = point_idx.tile((0, batch_size))
        
        idx = torch.stack([batch_idx, winning_vote_idx, point_idx], dim=-1)
        winning_vote = torch.index_select(self._votes, idx)
        winning_pres = torch.index_select(self._vote_presence_prob, idx)
        vote_presence = torch.gt(mixing_logits[:, :-1],
                               mixing_logits[:, -1:])
        is_from_capsule = winning_vote_idx // self._n_votes
        posterior_mixing_probs = F.softmax(posterior_mixing_logits_per_point,
                                           dim=1)
        ### UNFINISHED ###
        dummy_vote = torch.empty(self._votes[:1, :1].shape)
        nn.init.uniform_(dummy_vote)
        dummy_vote = dummy_vote.tile((0, batch_size))
        dummy_pres = torch.zeros((batch_size, 1, n_input_points))
        votes = torch.cat((self._votes, dummy_vote), 1)
        pres = torch.cat([self._vote_presence_prob, dummy_pres], 1)
    
        soft_winner = torch.sum((posterior_mixing_probs,
                                 -1).unsqueeze(1) * votes, 1)
        soft_winner_pres = torch.sum(
            posterior_mixing_probs * pres, dim=1)
    
        posterior_mixing_probs = torch.permute(
            posterior_mixing_probs[:, :-1], (0, 2, 1))
    
        assert winning_vote.shape == x.shape
    
        return self.OutputTuple(
            log_prob=mixture_log_prob_per_batch,
            vote_presence=vote_presence.type(torch.FloatTensor),
            winner=winning_vote,
            winner_pres=winning_pres,
            soft_winner=soft_winner,
            soft_winner_pres=soft_winner_pres,
            posterior_mixing_probs=posterior_mixing_probs,
            is_from_capsule=is_from_capsule,
            mixing_logits=mixing_logits,
            mixing_log_prob=mixing_log_prob,
        )

def _capsule_entropy(caps_presence_prob, k=1, **unused_kwargs):
    del unused_kwargs
    within_example = math.normalize(caps_presence_prob, 1)
    within_example = math.safe_ce(within_example, within_example*k)
    between_example = torch.sum(caps_presence_prob, 0)
    between_example = math.normalize(between_example, 0)
    between_example = math.safe_ce(between_example, between_example * k)
    return within_example, between_example


def _neg_capsule_kl(caps_presence_prob, **unused_kwargs):
    del unused_kwargs
    num_caps = int(caps_presence_prob.shape[-1])
    return _capsule_entropy(caps_presence_prob, k=num_caps)

def _caps_pres_l2(caps_presence_prob, num_classes=10.,
                  within_example_constant=0., **unused_kwargs):
    del unused_kwargs
    batch_size, num_caps = caps_presence_prob.shape.as_list()
    if within_example_constant == 0.:
        within_example_constant = float(num_caps) / num_classes
    between_example_constant = float(batch_size) / num_classes
    
    within_example = 0.5*torch.square(
      torch.sum(caps_presence_prob, 1)
      - within_example_constant) / batch_size * 2.
    between_example = 0.5*torch.square(
      torch.sum(caps_presence_prob, 0)
      - between_example_constant) / num_caps * 2.
    return within_example, -between_example


def sparsity_loss(loss_type, *args, **kwargs):
  if loss_type == 'entropy':
    sparsity_func = _capsule_entropy

  elif loss_type == 'kl':
    sparsity_func = _neg_capsule_kl

  elif loss_type == 'l2':
    sparsity_func = _caps_pres_l2

  else:
    raise ValueError(
        'Invalid sparsity loss: "{}"'.format(loss_type))

  return sparsity_func(*args, **kwargs)