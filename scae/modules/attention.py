import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SetTransformer(nn.Module):

    """Permutation invariant Transformer."""

    def __init__(self,
                 n_layers,
                 n_heads,
                 n_dims,
                 n_output_dims,
                 n_outputs,
                 layer_norm=False,
                 dropout_rate=0.,
                 n_inducing_points=0):
        super(SetTransformer, self).__init__()
        self._n_layers = n_layers
        self._n_heads = n_heads
        self._n_dim = n_dims
        self._n_output_dims = n_output_dims
        self._n_outputs = n_outputs
        self._layer_norm = layer_norm
        self._dropout_rate = dropout_rate
        self._n_inducing_points = n_inducing_points

    def forward(self, x, presence=None):

        batch_size = int(x.shape[0])

        # generic linear mlp applied over whole batch
        print("$$$$$$$$", x.shape)
        h = nn.Linear(23, self._n_dim)(x)

        args = [self._n_heads, self._layer_norm, self._dropout_rate]
        klass = SelfAttention

        # TODO: lets avoid induced attention for now on the encoder
        # if self._n_inducing_points > 0:
        #     args = [self._n_inducing_points] + args
        #     klass = InducedSelfAttention

        # From Paper: Encoder(X) := SAB(SAB(X)) : SAB = Self-Attention Block
        for _ in range(self._n_layers):
            h = klass(*args)(h, presence)
        z = nn.Linear(16, 16)(h)

        # From Paper: Decoder(Z) = rFF(SAB(PMA_k(Z)))
        # s.t. rFF is a row-wise (sample independent) Feed-Forward Layer
        # and PMA_k is Pooling by Multihead Attention induced on k seed vectors

        # TODO: avoid induced attention for now.
        # Unclear where tf global variable is coming from and what we want for
        # these values.

        # inducing_points = tf.get_variable(
        #     'inducing_points', shape=[1, self._n_outputs, self._n_output_dims])
        # inducing_points = snt.TileByDim([0], [batch_size])(inducing_points)

        return MultiHeadQKVAttention(self._n_heads)(z, z, z, presence)
        # return MultiHeadQKVAttention(self._n_heads)(inducing_points, z, z, presence)


class QKVAttention(nn.Module):

    """Generic QKV Attention module.

    Att(Q, K, V; w) := w(QK^T)V
    """

    def forward(self, queries, keys, values, presence=None):
        """
        :param queries: Tensor of shape [B, N, d_k].
        :param keys: Tensor of shape [B, M, d_k].
        :param values: Tensor of shape [B, M, d_v].
        :param presence: None or tensor of shape [B, M].

        :returns: Tensor of shape [B, N, d_v]
        """

        n_dim = int(queries.shape[-1])

        # [B, M, d] x [B, d, N] = [B, M, N]
        print("DJDN", queries.shape, keys.transpose(1, 2).shape)
        routing = torch.matmul(queries, keys.transpose(1, 2))

        if presence is not None:
            presence = torch.unsqueeze(presence, -2).float()
            routing -= (1. - presence) * 1e32

        routing = F.softmax(routing / np.sqrt(n_dim), -1)

        # every output is a linear combination of all inputs
        # [B, M, N] x [B, N, d_v] = [B, M, d_v]
        res = torch.matmul(routing, values)
        return res


class MultiHeadQKVAttention(nn.Module):

    """Adapted multi-head version of `QKVAttention` module.

    Multihead(Q, K, V; w) := concat(O_1, ..., O_h)W^0

    s.t. O_j = w(QK^T)V (Attention on reduced dimension of input)
    """

    def __init__(self, n_heads):
        super(MultiHeadQKVAttention, self).__init__()
        self._n_heads = n_heads

    def forward(self, queries, keys, values, presence=None):

        def transform(x, n=self._n_heads):
            n_dim = np.ceil(float(int(x.shape[-1])) / n)
            return nn.Linear(int(n_dim), int(n_dim))(x)

        outputs = []
        for _ in range(self._n_heads):
            args = [transform(i) for i in [queries, keys, values]]
            if presence is not None:
                args.append(presence)
            outputs.append(QKVAttention()(*args))

        linear = nn.Linear(values.shape[-1], values.shape[-1])
        return linear(torch.cat((outputs), dim=-1))


class SelfAttention(nn.Module):

    """SAB(X) := MAB(X, X)

    MAB(X, Y) := LayerNorm(H + rFF(H))
        : H = LayerNorm(X + Multihead(X, Y, Y; w)

    ie. Self-Attention Block is MultiheadAttention where X = Y. """

    def __init__(self, n_heads, layer_norm=False, dropout_rate=0.):
        super(SelfAttention, self).__init__()
        self._n_heads = n_heads
        self._layer_norm = layer_norm
        self._dropout_rate = dropout_rate

    def forward(self, x, presence=None):
        n_dims = int(x.shape[-1])

        y = self._self_attention(x, presence)

        if self._dropout_rate > 0.:
            x = nn.Dropout(self._dropout_rate)(x)

        y += x

        if presence is not None:
            y *= torch.unsqueeze(presence.float(), -1)

        if self._layer_norm:
            y = nn.LayerNorm(y[1:])(y)

        # Two layers of row-wise (independent) feed-forward
        h = nn.Linear(n_dims, 2*n_dims)(x)
        h = nn.Linear(2*n_dims, n_dims)(h)
        
        if self._dropout_rate > 0.:
            h = nn.Dropout(self._dropout_rate)(h)

        h += y

        if self._layer_norm:
            h = nn.LayerNorm(h[1:])(h)

        return h

    def _self_attention(self, x, presence):
        heads = MultiHeadQKVAttention(self._n_heads)
        return heads(x, x, x, presence)


class InducedSelfAttention(SelfAttention):

    """Self-attention with inducing points a.k.a. ISAB from SetTransformer.

    ISAB(X) := MAB(X, H) : H := MAB(I, X)
    Induced self-attention on kernel of reduced dimensionality for performance.

    TODO: IMPLEMENT GLOBAL VARIABLE, avoid induced attention for now (can get
    away with this for constellation/mnist
    """

    # def __init__(self, n_inducing_points, n_heads, layer_norm=False,
    #              dropout_rate=0.):
    #     super(InducedSelfAttention, self).__init__(n_heads, layer_norm,
    #                                                dropout_rate)

    #     self._n_inducing_points = n_inducing_points

    # def _self_attention(self, x, presence):

    #     head_before = MultiHeadQKVAttention(self._n_heads)
    #     # head_after = MultiHeadQKVAttention(self._n_heads)
    #     head_after = head_before

    #     inducing_points = tf.get_variable(
    #         'inducing_points', shape=[1, self._n_inducing_points,
    #                                   int(x.shape[-1])])

    #     inducing_points = snt.TileByDim(
    #         [0], [int(x.shape[0])])(inducing_points)

    #     z = head_before(inducing_points, x, x)
    #     y = head_after(x, z, z)
    #     return y
