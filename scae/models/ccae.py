from easydict import EasyDict

import torch
import torch.nn as nn
import pytorch_lightning as pl

import scae.util.math as math
import scae.modules.capsule as capsule


class CCAE(pl.LightningModule):

    """Reconstruct points from constellations (groupings of points).

    Two-dimensional points as "parts" in the SCAE paradigm. Object capsule
    behavior and reconstruction logic is nearly identical to OCAE."""

    def __init__(self, encoder, capsule, args):
        super(CCAE, self).__init__()

        self.encoder = encoder
        self.capsule = capsule

        self.lr = args.ccae.lr
        self.lr_decay = args.ccae.lr_decay
        self.weight_decay = args.ccae.weight_decay

        # self.n_classes = args.num_classes
        self.mse = nn.MSELoss()

    def forward(self, points, presence=None):
        """Predicting parts from object capsules + loglikelihood.

        Additional details here..."""

        h = self.encoder(points)
        res = self.capsule(h)

        #
        # Parse the capsule predictions for loglikelihood parameters.
        #

        batch_size, n_input_points = int(points.shape[0]), int(points.shape[0])
        # self.vote_shape = [batch_size, self._n_caps, self._n_votes, 6]

        res.vote = res.vote[..., :-1, -1]

        def pool_dim(x, dim_begin, dim_end):
            combined_shape = list(
                x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
            return x.view(combined_shape)

        for k, v in res.items():
            if k == "vote" or k == "scale":
                res[k] = pool_dim(v, 1, 3)
            if k == "vote_presence":
                res[k] = pool_dim(v, 1, 3)

        #
        # Parse the loglikelihood output to parameterize loss.
        #

        # TODO: pull "up" - _n_votes = 6
        self._n_votes = 4
        self._n_caps = 3

        likelihood = capsule.OrderInvariantCapsuleLikelihood(self._n_votes,
                                                             res.vote,
                                                             res.scale,
                                                             res.vote_presence)
        ll_res = likelihood(points, presence)

        #############
        #  ll prob  #
        #############

        soft_layer = torch.nn.Softmax(dim=1)
        mixing_probs = soft_layer(ll_res.mixing_logits)
        prior_mixing_log_prob = math.scalar_log(1. / n_input_points)
        mixing_kl = mixing_probs * \
            (ll_res.mixing_log_prob - prior_mixing_log_prob)
        mixing_kl = torch.mean(torch.sum(mixing_kl, -1))

        ##############
        #  sparsity  #
        ##############

        from_capsule = ll_res.is_from_capsule

        # torch implementation of tf.one_hot
        idx = torch.eye(self._n_caps)
        wins_per_caps = torch.stack([idx[from_capsule[b].type(
            torch.LongTensor)] for b in range(from_capsule.shape[0])])

        if presence is not None:
            wins_per_caps *= torch.expand_dims(presence, -1)

        wins_per_caps = torch.sum(wins_per_caps, 1)

        has_any_wins = torch.gt(wins_per_caps, 0).float()
        should_be_active = torch.gt(wins_per_caps, 1).float()

        # https://pytorch.org/docs/stable/generated/torch.nn.MultiLabelSoftMarginLoss.html
        # From math, looks to be same as `tf.nn.sigmoid_cross_entropy_with_logits`.

        # TODO: not rigorous cross-implementation
        softmargin_loss = torch.nn.MultiLabelSoftMarginLoss()
        sparsity_loss = softmargin_loss(should_be_active,
                                        res.pres_logit_per_caps)

        # sparsity_loss = tf.reduce_sum(sparsity_loss * has_any_wins, -1)
        # sparsity_loss = tf.reduce_mean(sparsity_loss)

        caps_presence_prob = torch.max(torch.reshape(
            res.vote_presence, [batch_size, self._n_caps, self._n_votes]),
            2)[0]

        return EasyDict(
            log_prob=ll_res.log_prob,
            vote=res.vote,
            mixing_kl=mixing_kl,
            sparsity_loss=sparsity_loss,
            caps_presence_prob=caps_presence_prob,
            mean_scale=torch.mean(res.scale)
        )

    # def training_step(self, batch, batch_idx):
    #     # TODO
    #     # Pulled from domas' code
    #     pass

    # def training_epoch_end(self, outputs):
    #     # TODO
    #     # Pulled from domas' code
    #     pass

    # def configure_optimizers(self):
    #     # TODO
    #     # Pulled from domas' code
    #     pass
