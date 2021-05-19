import random

import numpy as np
from easydict import EasyDict

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

import scae.util.math as math_utils

class ConvAttn(nn.Module):
    def __init__(self, n_caps, n_dims, in_channels=128):
        super(ConvAttn, self).__init__()
        self._n_caps = n_caps
        self._n_dims = n_dims
        self._in_channels = in_channels

        self._feat = nn.Conv2d(in_channels, n_dims * n_caps, kernel_size=1, stride=1)
        self._attn = nn.Conv2d(in_channels, n_caps, kernel_size=1, stride=1)

    def forward(self, x):
        """
        Compute and apply per-capsule attention masks
        :param x: Tensor of shape (batch_size, self._in_channels, H, W)
        :return: Tensor of shape (batch_size, self._n_caps, self._n_dims)
        """
        # TODO: add visualization of attn maps for debugging
        batch_size = x.shape[0]

        a = self._attn(x)
        a = a.reshape(batch_size, self._n_caps, 1, -1)  # flatten image dims (HxW)
        a_mask = nn.functional.softmax(a, dim=-1)

        f = self._feat(x)
        f = f.reshape(batch_size, self._n_caps, self._n_dims, -1)  # flatten image dims (HxW)

        # weighted sum over all image pixels, different weighting for each capsule
        return (f * a_mask).sum(-1, keepdim=False)


class GroupEqConv2d(nn.Module):
    # Implementation of Group Equivariant Convolutional Networks: https://arxiv.org/abs/1602.07576v3
    GROUPS = {
        "p4": (
            # Rot 0 CW = identity op
            lambda im: im,
            # Rot 90 CW = flip over x-axis, then transpose image
            lambda im: im.flip(-2).permute(0, 1, 3, 2),
            # Rot 90 CCW = transpose image, then flip over x-axis
            lambda im: im.permute(0, 1, 3, 2).flip(-2),
            # Rot 180 CW = flip over x-axis and y-axis
            lambda im: im.flip(-1).flip(-2)
        )
    }

    def __init__(self, in_channels, out_channels, kernel_size, group=GROUPS["p4"], **kwargs):
        super(GroupEqConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_size = len(group)
        self.group = group

        assert out_channels % self.group_size == 0, "Num output channels must be a multiple of group size"
        self.conv = nn.Conv2d(in_channels, out_channels // self.group_size, kernel_size, **kwargs)
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, x):
        batch_size = x.shape[0]
        x_transforms = torch.cat([f(x) for f in self.group])
        z = self.conv(x_transforms)
        # z.reshape(batch_size, self.group_size, self.out_channels // self.group_size, *z.shape[-2:])
        return z.reshape(batch_size, -1, *z.shape[-2:])
        # return z.reshape(batch_size, self.group_size, self.out_channels, *z.shape[-2:]).mean(dim=1)



class CapsuleImageEncoder(nn.Module):
    def __init__(self, args):
        super(CapsuleImageEncoder, self).__init__()
        self._n_caps = args.pcae.num_caps
        self._caps_dim = args.pcae.caps_dim
        self._feat_dim = args.pcae.feat_dim
        self._noise_scale = args.pcae.encoder.noise_scale
        self._inverse_space_transform = args.pcae.encoder.inverse_space_transform

        # Image embedding encoder
        # channels = [args.im_channels, 128, 128, 128, 128]
        channels = [args.im_channels, 32, 32, 32, 128]
        strides = [2, 2, 1, 1]
        layers = []
        for i in range(4):
            # layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=strides[i]))
            layers.append(GroupEqConv2d(channels[i], channels[i + 1], kernel_size=3, stride=strides[i]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(channels[i+1]))
        self._encoder = nn.Sequential(*layers)

        # Conv attention
        self._splits = [self._caps_dim, self._feat_dim, 1]  # 1 for presence
        self._n_dims = sum(self._splits)
        self._attn = ConvAttn(self._n_caps, self._n_dims)

    def forward(self, x):
        """

        :param x: Image tensor of shape (batch_size, 1, H, W)
        :return:
        """
        batch_size = x.shape[0]
        img_embedding = self._encoder(x)  # img_embedding shape (batch_size, C, H, W)
        preds = self._attn(img_embedding)  # preds shape (batch_size, self._n_caps, self._n_dims)

        poses, features, presence_logits = preds.split(self._splits, dim=-1)

        # Tensor of shape (batch_size, self._n_caps, 6)
        poses = math_utils.geometric_transform(poses, True, inverse=self._inverse_space_transform)

        if self._feat_dim == 0:
            features = None

        presence_logits = presence_logits.squeeze(-1)
        if self._noise_scale > 0.:  # TODO: why do this???
            # Add uniform [-self._noise_scale/2, self._noise_scale/2] noise to logits
            presence_logits = presence_logits + ((torch.rand(presence_logits.shape).cuda() - .5) * self._noise_scale)
        presences = torch.sigmoid(presence_logits)

        return EasyDict(
            poses=poses,
            features=features,
            presences=presences,
            presence_logits=presence_logits,
            img_embedding=img_embedding
        )


def get_nonlin(name):
    nonlin = getattr(torch, name, None)
    if nonlin:
        return nonlin
    else:
        raise ValueError('Invalid nonlinearity: "{}".'.format(name))


class TemplateImageDecoder(nn.Module):
    def __init__(self, args):
        super(TemplateImageDecoder, self).__init__()
        self._n_caps = args.pcae.num_caps
        self._output_size = args.pcae.decoder.output_size
        self._template_size = args.pcae.decoder.template_size
        self._n_channels = args.im_channels
        self._colorize_templates = False

        self._template_nonlin = get_nonlin(args.pcae.decoder.template_nonlin)
        self._color_nonlin = get_nonlin(args.pcae.decoder.color_nonlin)
        self._use_alpha_channel = args.pcae.decoder.alpha_channel

        assert len(self._template_size) == 2, 'Template size must be of dim 2'
        self.init_templates()
        self.bg_value = torch.nn.Parameter(torch.tensor([0.]), requires_grad=True)
        self.bg_logit = torch.nn.Parameter(torch.tensor([0.]), requires_grad=True)

    def init_templates(self):
        template_shape = [self._n_caps, self._n_channels] + list(self._template_size)  # torch generally uses (N, C, H, W)
        n_elems = np.prod(template_shape[1:])

        # make each templates orthogonal to each other at init
        n = max(self._n_caps, n_elems)
        q, _ = torch.qr(torch.rand(n, n))  # TODO: test whether orthogonal init is even helpful

        col_idxs = list(range(q.shape[1]))
        random.shuffle(col_idxs)
        q = q[:, torch.tensor(col_idxs)]

        ts = q[:self._n_caps, :n_elems].reshape(template_shape)

        t_min = ts.min()
        t_max = ts.max()
        ts = (ts - t_min) / (t_max - t_min) * 2 - 1  # normalize values to [-1,1]
        if self._use_alpha_channel:
            alphas = torch.zeros(self._n_caps, 1, *self._template_size)
            ts = torch.cat([ts, alphas], dim=1)
        else:
            self.temperature_logit = torch.nn.Parameter(torch.tensor([0.]), requires_grad=True)

        self.templates = torch.nn.Parameter(ts, requires_grad=True)

    def forward(self, poses, presences=None):
        """

        :param capsules:
        :param bg_image: size (N, C, H, W)
        :return:
        """
        batch_size = poses.shape[0]
        n_dims = self._n_channels + 1 if self._use_alpha_channel else self._n_channels

        template_batch_shape = [batch_size, self._n_caps, n_dims] + list(self._output_size)

        # poses shape (batch_size * self._n_caps, 2, 3)
        poses = poses.view(-1, 2, 3)
        # TODO: port to using https://kornia.readthedocs.io/en/latest/geometry.transform.html#kornia.geometry.transform.warp_affine
        grid_coords = nn.functional.affine_grid(theta=poses, size=(poses.shape[0], n_dims, *self._output_size))

        # templates             shape             (self._n_caps, n_dims, self._template_size)
        # template_stack        shape (batch_size* self._n_caps, n_dims, self._template_size)
        # transformed_templates shape (batch_size, self._n_caps, n_dims, self._output_size)
        templates = self._template_nonlin(self.templates)
        template_stack = templates.repeat(batch_size, 1, 1, 1)   # TODO: see if auto broadcasting over batch dim works
        transformed_templates = nn.functional.grid_sample(template_stack, grid_coords).view(template_batch_shape)

        bg_value = torch.sigmoid(self.bg_value)
        bg_image = torch.zeros(batch_size, 1, self._n_channels, *self._output_size).cuda() + bg_value

        # presences          shape (batch_size, self._n_caps)
        presence_probs = presences.view(batch_size, self._n_caps, 1, 1, 1)

        if self._use_alpha_channel:
            tt_rgb, tt_a = transformed_templates.split((self._n_channels, 1), dim=2)
            # template_logits    shape (batch_size, self._n_caps, self._output_size)
            tt_logits = tt_a + math_utils.safe_log(presence_probs)
            bg_logits = self.bg_logit
        else:
            tt_rgb = transformed_templates

            temperature = F.softplus(self.temperature_logit + .5) + 1e-4
            # TODO: why is this improper logit addition good for training?
            tt_logits = tt_rgb / temperature + math_utils.safe_log(presence_probs)
            # tt_logits = -F.relu(-tt_logits)  # ensure logits are negative
            # tt_logits = math_utils.safe_log(presence_probs).expand_as(tt_rgb)
            bg_logits = bg_image / temperature

        bg_logits = bg_logits.expand(batch_size, 1, 1, *self._output_size)
        # TODO: add template colorization from features

        # mixture_logits shape (batch_size, self._n_caps + 1, self._n_channels, self._output_size)
        # mixture_means  shape (batch_size, self._n_caps + 1, self._n_channels, self._output_size)
        mixture_logits = torch.cat([tt_logits, bg_logits], dim=1)
        mixture_means = torch.cat([tt_rgb, bg_image], dim=1)
        mixture_pdf = math_utils.MixtureDistribution(mixture_logits, mixture_means)

        return EasyDict(
            raw_templates=templates,
            mixture_means=mixture_means,
            mixture_logits=mixture_logits,
            pdf=mixture_pdf,
        )

