from collections import OrderedDict
import collections
import torch.distributions as D
import torch.nn.functional as F
import scae.util.math as math
import numpy as np
import torch.nn as nn
import torch
from attrdict import AttrDict
from easydict import EasyDict


class CapsuleLayer(nn.Module):
    _n_transform_params = 6

    def __init__(self, input_dims, n_caps, n_caps_dims, n_votes,
                 n_caps_params=32,
                 n_hidden_dim=128, learn_vote_scale=False, deformations=True,
                 noise_type=None, noise_scale=0., similarity_transform=True,
                 caps_dropout_rate=0.0):
        """Instantiates a set of capsules.


        :param input_dims: The dimension of the input part (ie. a 2d point = 2)
        :param n_caps: The number of capsules included in the layer.
        :param n_caps_dims: The number of hidden layers in each capsule's MLP.
        :param n_votes: How many part votes each capsule gets.
        :param n_caps_params: Dimensionality of capsule feature vector.
        :param n_hidden_dim: Dimensionality of capsule hidden layers.
        :param learn_vote_scale:
        :param deformations:
        :param noise_type:
        :param noise_scale:
        :param similarity_transform:
        :param caps_dropout_rate:

        """
        super(CapsuleLayer, self).__init__()

        self.input_dims = input_dims  # last dimension of encoded feature
        self._n_caps = n_caps
        self._n_caps_dims = n_caps_dims
        self._n_caps_params = n_caps_params

        self._n_votes = n_votes
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

        self.caps_mlp_design = []
        shape_list = [self._n_caps_params+1, self._n_caps_dims, self.n_outputs]
        for i in range(1, len(shape_list)):
            self.caps_mlp_design.append(nn.Linear(shape_list[i-1],
                                                  shape_list[i], bias=False))
            self.caps_mlp_design.append(nn.ReLU())
        self.caps_mlp = [nn.Sequential(*self.caps_mlp_design) for i in
                         range(self._n_caps)]

    def forward(self, x, parent_transform=None, parent_presence=None):
        """Decodes part parameters from a set of encoded capsule features.

        Part parameters consist of:

            * A presence probability
            * A scale (standard dev.)
            * An object-part (OP) transformation matrix

        These parameters are then used to construct a "vote" by composing the
        OP matrix with the capsule's OV matrix.

        This "vote" is used along with its associated scale and presence
        probability to generate a likelihood score given a part.

        :param x: A set of capsules (represented as feature vectors)
            [B, #capsules, 32]

        :returns:
            :param vote: A matrix for each capsule's votes describing a part's pose.
                [B, #capsules, #votes, #OV*OP matrix ie. (3, 3)]
            :param scale: A std. dev. for each capsule's votes (for Gaussian
            mixture component).
                [B, #capsules, #votes]
            :param vote_presence: A probability representing confidence in vote.
                [B, #capsules, #votes]
            :param pres_logit_per_caps:
                [B, #capsules]
            :param pres_logit_per_vote:
                [B, #capsules, #votes]
            :param dynamic_weights_l2:
                [B, #capsules, #votes, 6]
            :param raw_caps_params:
                [B, #capsules, 39]
            :param raw_caps_features: The feature vector representing a capsule
            that is fed into the MLP.
                [B, #capsules, 32]

('pres_logit_per_caps', torch.Size([1, 3, 1])), ('pres_logit_per_vote', torch.Size([1, 3, 4])), ('dynamic_weights_l2', torch.Size([1, 3, 4, 6])), ('raw_caps_params', torch.Size([1, 3, 39])), ('raw_caps_features', torch.Size([1, 3, 32]))]
        """

        batch_size = x.shape[0]
        batch_shape = [batch_size, self._n_caps]

        #
        # Manually implement dropout by adding layer sampled from Bernoulli.
        #

        if self._caps_dropout_rate == 0.0:
            caps_exist = torch.ones(batch_shape + [1], dtype=torch.float32)
        else:
            pmf = D.bernoulli.Bernoulli(1. - self._caps_dropout_rate,
                                        dtype=torch.float32)
            caps_exist = pmf.sample(batch_shape + [1])

        caps_params = torch.cat([x, caps_exist], -1)
        split = torch.split(caps_params, 1, 1)
        all_params = [self.caps_mlp[i](x) for i, x in enumerate(split)]
        raw_caps_params = torch.cat(all_params, 1)
        all_params = torch.split(raw_caps_params, self.splits, -1)

        #
        # Reshape our capsule outputs into the features we know and love:
        #   * cpr - "capsule pose relationship?" - pretty sure this is object
        #            part
        #   * ccr - "capsule capsule relationship?" - pretty sure this is
        #            object-viewpoint
        #   * pres_logit_per_caps - presence probability for capsule.
        #   * pres_logit_per_vote - presence probaility for vote.
        #   * scale_per_vote - scale (std dev.) for vote.
        #

        res = [torch.reshape(i, batch_shape + s)
               for (i, s) in zip(all_params, self.output_shapes)]

        cpr_dynamic = res[0]

        # ADD BIASES
        # res = [snt.AddBias()(i) for i in res[1:]]

        _, ccr, pres_logit_per_caps, pres_logit_per_vote, scale_per_vote = res

        if self._caps_dropout_rate != 0.0:
            pres_logit_per_caps += math.safe_log(caps_exist)

        # WTF IS CPR STATIC AND WHY DOES IT EXIST, all 0s...
        cpr_static = torch.zeros(size=[1, self._n_caps, self._n_votes,
                                       self._n_transform_params],
                                 requires_grad=True)
        pres_logit_per_caps = math.add_noise(pres_logit_per_caps)
        pres_logit_per_vote = math.add_noise(pres_logit_per_vote)

        #
        # Manipulating the ccr (OV) transform.
        #

        if parent_transform is None:
            ccr = math.geometric_transform(ccr, as_3x3=True, similarity=True)
        else:
            ccr = parent_transform
        if not self._deformations:
            cpr_dynamic = torch.zeros_like(cpr_dynamic)

        cpr = math.geometric_transform(cpr_dynamic + cpr_static, as_3x3=True,
                                       similarity=True)

        # FIX THIS REPEAT ERROR
        # ccr_per_vote = ccr.repeat((1,1,self._n_votes,1,1))

        # votes = torch.matmul(ccr_per_vote, cpr)

        #
        # The actual composition of object-viewpoint (ccr) and object-part
        # (cpr) transforms
        #

        votes = torch.matmul(ccr, cpr)

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

        # Gaussian pdfs generated from our the outputs of our capsules.
        self.expanded_votes = self._votes.unsqueeze(dim=1)
        self.expanded_scale = self._scales.unsqueeze(dim=1).unsqueeze(dim=-1)
        self.vote_component_pdf = self._get_pdf(self.expanded_votes,
                                                self.expanded_scale)

        self.mixing_logits = math.safe_log(self._vote_presence_prob)

    def forward(self, x, presence=None):
        """Compute the likelihood of a set of parts given some capsule
        predictions.

        :param x: A set of parts
            [B, #parts, part_dim]

        :returns:
            :param log_prob: Summed across batch dimension
                [1]
            :param vote_presence:
                [B, #capsules * #votes]
            :param winner: The vote, for each part, that maximizes
                likelihood.
                [B, #parts, #part_dim]
            :param winning_pres: The presence associated with winning vote.
                [B, #parts]
            :param is_from_capsule: The capsule associated with winning vote.
                [B, #parts]
            :param mixing_logits: Raw log probabilites from capsule decoding.
                [B, #capsules * #votes + 1]
            :param mixing_log_prob: Proper probabilities - normalized by vote
                dimension.
                [B, #capsules * #votes + 1]
            :param posterior_mixing_probs: Loglikelihoods for a part against
                all possible votes, weighted by mixing presence, and softmaxed.
                [B, #parts, #capsules * #votes]
        """

        self.batch_size, self.n_input_points = x.shape[0], x.shape[1]
        expanded_x = x.unsqueeze(dim=2)

        # Calculate log probs for each point/part wrt. a capsules vote.
        vote_log_prob_per_dim = self.vote_component_pdf.log_prob(expanded_x)
        vote_log_prob = torch.sum(vote_log_prob_per_dim, dim=-1)

        dummy_vote_log_prob = torch.zeros(
            [self.batch_size, self.n_input_points, 1])
        dummy_vote_log_prob -= 2. * math.scalar_log(10.)
        vote_log_prob = torch.cat([vote_log_prob, dummy_vote_log_prob], 2)

        mixing_logits = math.safe_log(self._vote_presence_prob)
        dummy_logit = torch.zeros(
            (self.batch_size, 1)) - 2. * math.scalar_log(10.)
        mixing_logits = torch.cat([mixing_logits, dummy_logit], 1)

        mixing_log_prob = mixing_logits - torch.logsumexp(mixing_logits, dim=1,
                                                          keepdims=True)

        expanded_mixing_logits = mixing_log_prob.unsqueeze(dim=1)
        mixture_log_prob_per_component = torch.logsumexp(
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

        batch_idx = torch.range(1, self.batch_size,
                                dtype=torch.int64).unsqueeze(dim=-1)
        batch_idx = batch_idx.repeat((1, winning_vote_idx.shape[-1]))

        idx = torch.stack([batch_idx, winning_vote_idx], dim=-1)

        # pytorch implementation of tf.gather_nd
        winning_vote = torch.ones(idx.shape)
        winning_pres = torch.ones(idx.shape[:-1])
        for b in range(idx.shape[0]):
            for v in range(idx.shape[1]):
                i, j = idx[b, v]
                i -= 1  # correct batch idx
                winning_vote[b, v] = self._votes[i, j]
                winning_pres[b, v] = self._vote_presence_prob[i, j]

        vote_presence = torch.gt(mixing_logits[:, :-1],
                                 mixing_logits[:, -1:])

        # the first four votes belong to the square
        is_from_capsule = winning_vote_idx // self._n_votes

        posterior_mixing_probs = F.softmax(
            posterior_mixing_logits_per_point, dim=-1)[..., :-1]

        assert winning_vote.shape == x.shape

        return EasyDict(
            log_prob=mixture_log_prob_per_batch,
            vote_presence=vote_presence.type(torch.FloatTensor),
            winner=winning_vote,
            winner_pres=winning_pres,
            is_from_capsule=is_from_capsule,
            mixing_logits=mixing_logits,
            mixing_log_prob=mixing_log_prob,
            # soft_winner=torch.zeros_like(winning_vote),
            # soft_winner_pres=torch.zeros_like(winning_pres),
            posterior_mixing_probs=posterior_mixing_probs,
        )

    def _get_pdf(self, votes, scale):
        if self._pdf == 'normal':
            pdf = D.normal.Normal(votes, scale)
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
        batch_size, n_input_points = list(x.shape[:2])
        expanded_x = x.unsqueeze(dim=1)
        vote_log_prob_per_dim = self.vote_component_pdf.log_prob(expanded_x)
        vote_log_prob = torch.sum(vote_log_prob_per_dim, dim=-1)
        dummy_vote_log_prob = torch.zeros([batch_size, 1, n_input_points])
        # log(10) being used
        dummy_vote_log_prob -= 2. * 2.30258
        vote_log_prob = torch.cat([vote_log_prob, dummy_vote_log_prob], dim=1)
        mixing_logits = math.safe_log(self._vote_presence_prob)
        dummy_logit = torch.zeros([batch_size, 1, 1]) - 2. * 2.30258
        dummy_logit = dummy_logit.repeat((1, 1, n_input_points))

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
        batch_idx = torch.range(0, batch_size-1,
                                dtype=torch.int64).unsqueeze(dim=1)
        batch_idx = batch_idx.repeat((1, n_input_points))
        point_idx = torch.range(0, n_input_points-1,
                                dtype=torch.int64).unsqueeze(dim=0)
        point_idx = point_idx.repeat((batch_size, 1))
        idx = torch.stack([batch_idx, winning_vote_idx, point_idx], dim=-1)

        # winning_vote = torch.index_select(self._votes, idx)
        # winning_pres = torch.index_select(self._vote_presence_prob, idx)

        # PERMUTE WAS USED AS A DIRTY WORKAROUND
        winning_vote = self._votes[list(idx.T)].permute(1, 0, 2)
        winning_pres = self._vote_presence_prob[list(idx.T)]
        vote_presence = torch.gt(mixing_logits[:, :-1],
                                 mixing_logits[:, -1:])
        is_from_capsule = winning_vote_idx // self._n_votes
        posterior_mixing_probs = F.softmax(posterior_mixing_logits_per_point,
                                           dim=1)
        ### UNFINISHED ###
        dummy_vote = torch.empty(self._votes[:1, :1].shape)
        nn.init.uniform_(dummy_vote)
        dummy_vote = dummy_vote.repeat((batch_size, 1, 1, 1))
        dummy_pres = torch.zeros((batch_size, 1, n_input_points))
        votes = torch.cat((self._votes, dummy_vote), 1)
        pres = torch.cat([self._vote_presence_prob, dummy_pres], 1)

        soft_winner = torch.sum(posterior_mixing_probs.unsqueeze(-1) * votes,
                                dim=1)
        soft_winner_pres = torch.sum(
            posterior_mixing_probs * pres, dim=1)

        posterior_mixing_probs = posterior_mixing_probs[:, :-1].permute(
            0, 2, 1)
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
