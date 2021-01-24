from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import timeit
import sys

import collections
from monty.collections import AttrDict
import numpy as np
import sonnet as snt
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import nest
import tensorflow_probability as tfp

from stacked_capsule_autoencoders.capsules import math_ops
from stacked_capsule_autoencoders.capsules.neural import BatchMLP

tfd = tfp.distributions


"""Permutation-invariant capsule layer useed in constellation experiments."""

OutputTuple = collections.namedtuple('CapsuleLikelihoodTuple',  # pylint:disable=invalid-name
                                     ('log_prob vote_presence winner '
                                      'winner_pres is_from_capsule '
                                      'mixing_logits mixing_log_prob '
                                      'soft_winner soft_winner_pres '
                                      'posterior_mixing_probs'))

# Below dictionary relevant for book keeping:
# self._capsule_kwargs {'n_caps_params': 32, 'n_hiddens': 128,
# 'learn_vote_scale': True, 'deformations': True, 'noise_type': 'uniform',
# 'noise_scale': 4, 'similarity_transform': True}


#
# CAPSULE PARAMETERS
#

_n_caps = 3
_n_votes = 4
_n_caps_dims = 2

#
# CAPSULE DECODINGS RES. (log likelihood parameterization)
#

# Vote (result of decoding and transformed)
# key: vote, value: Tensor("constellation_autoencoder/constellation_capsule/strided_slice:0", shape=(32, 3, 4, 2), dtype=float32)
_votes = tf.random.normal((32, 12, 2))

# Scale ~ std. dev
# key: scale, value: Tensor("constellation_autoencoder/constellation_capsule/capsule_layer/add_14:0", shape=(32, 3, 4), dtype=float32)
_scales = tf.random.normal((32, 12))

# (conditional) vote probability
# key: vote_presence, value: Tensor("constellation_autoencoder/constellation_capsule/capsule_layer/mul_18:0", shape=(32, 3, 4), dtype=float32)
_vote_presence_prob = tf.random.uniform((32, 12))


def _get_pdf(votes, scale):
    return tfd.Normal(votes, scale)

# Inputs
# *x* shape=(32, 11, 2), dtype=float32)
# *presence* shape=(32, 11), dtype=float32)


def naive_log_likelihood(x, presence=None):
    """Implementation from original repo ripped wholesale"""

    batch_size, n_input_points = x.shape[:2].as_list()

    # Generate gaussian mixture pdfs...
    # [B, 1, n_votes, n_input_dims]
    expanded_votes = tf.expand_dims(_votes, 1)
    expanded_scale = tf.expand_dims(tf.expand_dims(_scales, 1), -1)
    vote_component_pdf = _get_pdf(expanded_votes, expanded_scale)

    # For each part, evaluates all capsule, vote mixture likelihoods
    # [B, n_points, n_caps x n_votes, n_input_dims]
    expanded_x = tf.expand_dims(x, 2)
    vote_log_prob_per_dim = vote_component_pdf.log_prob(expanded_x)

    # Compressing mixture likelihood across all part dimension (ie. 2d point)
    # [B, n_points, n_caps x n_votes]
    vote_log_prob = tf.reduce_sum(vote_log_prob_per_dim, -1)
    dummy_vote_log_prob = tf.zeros([batch_size, n_input_points, 1])
    dummy_vote_log_prob -= 2. * tf.log(10.)
    # adding extra [B, n_points, n_caps x n_votes] to end. WHY?
    vote_log_prob = tf.concat([vote_log_prob, dummy_vote_log_prob], 2)

    # [B, n_points, n_caps x n_votes]
    # CONDITIONAL LOGIT a_(k,n)
    mixing_logits = math_ops.safe_log(_vote_presence_prob)

    dummy_logit = tf.zeros([batch_size, 1]) - 2. * tf.log(10.)
    mixing_logits = tf.concat([mixing_logits, dummy_logit], 1)

    #
    # Following seems relevant only towards compressing ll for loss.
    # REDUNDANCY
    #

    # mixing_logits -> presence (a)
    # vote_log_prob -> Gaussian value (one per vote) for each coordinate

    # BAD -> vote presence / summed vote presence
    mixing_log_prob = mixing_logits - tf.reduce_logsumexp(mixing_logits, 1,
                                                          keepdims=True)

    # BAD -> mixing presence (above) * each vote gaussian prob
    expanded_mixing_logits = tf.expand_dims(mixing_log_prob, 1)
    # Reduce to loglikelihood given k,n combination (capsule, vote)
    mixture_log_prob_per_component\
        = tf.reduce_logsumexp(expanded_mixing_logits + vote_log_prob, 2)

    if presence is not None:
        presence = tf.to_float(presence)
        mixture_log_prob_per_component *= presence

    # Reduce votes to single capsule
    # ^ Misleading, reducing across all parts, multiplying log
    # likelihoods for each part _wrt all capsules_.
    mixture_log_prob_per_example\
        = tf.reduce_sum(mixture_log_prob_per_component, 1)

    # Same as above but across all compressed part likelihoods in a batch.
    mixture_log_prob_per_batch = tf.reduce_mean(
        mixture_log_prob_per_example)

    #
    # Back from compression to argmax (routing to proper k)
    #

    # [B, n_points, n_votes]
    posterior_mixing_logits_per_point = expanded_mixing_logits + vote_log_prob
    # [B, n_points]
    winning_vote_idx = tf.argmax(
        posterior_mixing_logits_per_point[:, :, :-1], 2)

    batch_idx = tf.expand_dims(tf.range(batch_size, dtype=tf.int64), -1)
    batch_idx = snt.TileByDim([1], [winning_vote_idx.shape[-1]])(batch_idx)

    idx = tf.stack([batch_idx, winning_vote_idx], -1)
    winning_vote = tf.gather_nd(_votes, idx)
    winning_pres = tf.gather_nd(_vote_presence_prob, idx)
    vote_presence = tf.greater(mixing_logits[:, :-1],
                               mixing_logits[:, -1:])

    # the first four votes belong to the square
    # Just assuming the votes are ordered by capsule...
    is_from_capsule = winning_vote_idx // _n_votes

    posterior_mixing_probs = tf.nn.softmax(
        posterior_mixing_logits_per_point, -1)[Ellipsis, :-1]

    assert winning_vote.shape == x.shape

    return OutputTuple(
        log_prob=mixture_log_prob_per_batch,
        vote_presence=tf.to_float(vote_presence),
        winner=winning_vote,
        winner_pres=winning_pres,
        is_from_capsule=is_from_capsule,
        mixing_logits=mixing_logits,
        mixing_log_prob=mixing_log_prob,
        # TODO(adamrk): this is broken
        soft_winner=tf.zeros_like(winning_vote),
        soft_winner_pres=tf.zeros_like(winning_pres),
        posterior_mixing_probs=posterior_mixing_probs,
    )


def argmax_log_likelihood(x, presence=None):
    """Most simple of the optimization schemes.

    Skip the product of closeform probability of part given _all_ data. Rather
    use the value at the argmax as a proxy for each part.
    """

    batch_size, n_input_points = x.shape[:2].as_list()

    # Generate gaussian mixture pdfs...
    # [B, 1, n_votes, n_input_dims]
    expanded_votes = tf.expand_dims(_votes, 1)
    expanded_scale = tf.expand_dims(tf.expand_dims(_scales, 1), -1)
    vote_component_pdf = _get_pdf(expanded_votes, expanded_scale)

    # For each part, evaluates all capsule, vote mixture likelihoods
    # [B, n_points, n_caps x n_votes, n_input_dims]
    expanded_x = tf.expand_dims(x, 2)
    vote_log_prob_per_dim = vote_component_pdf.log_prob(expanded_x)

    # Compressing mixture likelihood across all part dimension (ie. 2d point)
    # [B, n_points, n_caps x n_votes]
    vote_log_prob = tf.reduce_sum(vote_log_prob_per_dim, -1)
    dummy_vote_log_prob = tf.zeros([batch_size, n_input_points, 1])
    dummy_vote_log_prob -= 2. * tf.log(10.)
    # adding extra [B, n_points, n_caps x n_votes] to end. WHY?
    vote_log_prob = tf.concat([vote_log_prob, dummy_vote_log_prob], 2)

    # [B, n_points, n_caps x n_votes]
    # CONDITIONAL LOGIT a_(k,n)
    mixing_logits = math_ops.safe_log(_vote_presence_prob)

    dummy_logit = tf.zeros([batch_size, 1]) - 2. * tf.log(10.)
    mixing_logits = tf.concat([mixing_logits, dummy_logit], 1)

    # BAD -> vote presence / summed vote presence
    mixing_log_prob = mixing_logits - tf.reduce_logsumexp(mixing_logits, 1,
                                                          keepdims=True)

    expanded_mixing_logits = tf.expand_dims(mixing_log_prob, 1)

    # [B, n_points, n_votes]
    posterior_mixing_logits_per_point = expanded_mixing_logits + vote_log_prob
    # [B, n_points]
    winning_vote_idx = tf.argmax(
        posterior_mixing_logits_per_point[:, :, :-1], 2)

    batch_idx = tf.expand_dims(tf.range(batch_size, dtype=tf.int64), -1)
    batch_idx = snt.TileByDim([1], [winning_vote_idx.shape[-1]])(batch_idx)

    idx = tf.stack([batch_idx, winning_vote_idx], -1)
    winning_vote = tf.gather_nd(_votes, idx)
    winning_pres = tf.gather_nd(_vote_presence_prob, idx)
    vote_presence = tf.greater(mixing_logits[:, :-1],
                               mixing_logits[:, -1:])

    # the first four votes belong to the square
    # Just assuming the votes are ordered by capsule...
    is_from_capsule = winning_vote_idx // _n_votes

    posterior_mixing_probs = tf.nn.softmax(
        posterior_mixing_logits_per_point, -1)[Ellipsis, :-1]

    assert winning_vote.shape == x.shape

    # log_prob=mixture_log_prob_per_batch,
    return OutputTuple(
        log_prob=None,
        vote_presence=tf.to_float(vote_presence),
        winner=winning_vote,
        winner_pres=winning_pres,
        is_from_capsule=is_from_capsule,
        mixing_logits=mixing_logits,
        mixing_log_prob=mixing_log_prob,
        # TODO(adamrk): this is broken
        soft_winner=tf.zeros_like(winning_vote),
        soft_winner_pres=tf.zeros_like(winning_pres),
        posterior_mixing_probs=posterior_mixing_probs,
    )


#
# CAPSULE ENCODING RES. (log likelihood input)
#

# parts / points
# x Tensor("IteratorGetNext:0", shape=(32, 11, 2), dtype=float32)
x = tf.random.uniform((32, 11, 2))

# object probability
# presence Tensor("IteratorGetNext:3", shape=(32, 11), dtype=float32)
presence = tf.random.uniform((32, 11))


naive = timeit.timeit(lambda: naive_log_likelihood(
    x, presence), number=100) / 100

argmax = timeit.timeit(lambda: argmax_log_likelihood(
    x, presence), number=100) / 100

print(
    f'Naive evaluation of synthetic loglikelihood averaged over 100 runs: {naive}')
print(
    f'Strictly argmaxed evaluation of synthetic loglikelihood averaged over 100 runs: {argmax}')
