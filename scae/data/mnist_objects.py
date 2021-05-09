import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib as pth
from torchvision.datasets import CIFAR10, MNIST, QMNIST, USPS
from torchvision import transforms
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from easydict import EasyDict

from scae.models.pcae import PCAE
from scae.modules.part_capsule_ae import TemplateImageDecoder
from scae.args import parse_args
import scae.util.math as math_utils
from scae.util.vis import plot_image_tensor_2D, plot_image_tensor

class MNISTObjects(torch.utils.data.Dataset):
    NUM_SAMPLES = 10000
    NUM_CLASSES = 10

    def __init__(self, root='data', train=True,
                 template_src='mlatberkeley/StackedCapsuleAutoEncoders/67lzaiyq',
                 num_caps=4, new=True, aligned=True, template_mixing='pdf'):
        self.train = train
        self.num_caps = num_caps
        self.file = pth.Path(root) / 'mnist_objects.pkl'
        # if self.file.exists() and not new:
        #     with open(self.file, 'rb') as f:
        #         self.data = pickle.load(f)
        # else:
        #     with open(self.file, 'wb') as f:
        #         self._generate(template_src)
        #         pickle.dump(self.data, f)

        self.aligned = aligned
        self.template_mixing = template_mixing
        self._generate(template_src)
        # self.plot(100)

    def _generate(self, template_src):
        with torch.no_grad():
            args = parse_args(f'--cfg scae/config/mnist.yaml --debug'.split(' '))
            args.pcae.num_caps = self.num_caps
            args.im_channels = 1

            pcae_decoder = TemplateImageDecoder(args).cuda()
            if template_src is not None:
                import wandb
                from pytorch_lightning.utilities.cloud_io import load as pl_load
                best_model = wandb.restore('last.ckpt', run_path=template_src, replace=True)
                pcae_decoder.templates = torch.nn.Parameter(pl_load(best_model.name)['state_dict']['decoder.templates'].contiguous())
            else:
                return NotImplementedError("Dataset templates need to be fetched from wandb run")
            self.templates = pcae_decoder._template_nonlin(pcae_decoder.templates)

            valid_part_poses = []
            valid_presences = []
            while len(valid_part_poses) < MNISTObjects.NUM_CLASSES:
                presences_shape = (MNISTObjects.NUM_CLASSES, self.num_caps)
                presences = Bernoulli(.99).sample(presences_shape).float().cuda()

                part_poses = self.rand_poses((MNISTObjects.NUM_CLASSES, self.num_caps),
                                             size_ratio=args.pcae.decoder.template_size[0] / args.pcae.decoder.output_size[0] / 2)
                part_poses = math_utils.geometric_transform(part_poses, similarity=True, inverse=True, as_matrix=True)

                temp_poses = part_poses[..., :2, :]
                temp_poses = temp_poses.reshape(*temp_poses.shape[:-2], 6)

                transformed_templates = self.transform_templates(self.templates, temp_poses)
                metric = self.overlap_metric(transformed_templates, presences)
                metric = metric * (presences.bool().unsqueeze(-1) | presences.bool().unsqueeze(-2)).float()

                for i in range(MNISTObjects.NUM_CLASSES):
                    if ((metric[i] == 0) | ((10 < metric[i]) & (metric[i] < 20))).all()\
                            and (metric[i] > 0).any():
                        valid_part_poses.append(part_poses[i])
                        valid_presences.append(presences[i])

            part_poses = torch.stack(valid_part_poses[:MNISTObjects.NUM_CLASSES])
            presences = torch.stack(valid_presences[:MNISTObjects.NUM_CLASSES])

            # Vis final objects
            # temp_poses = part_poses[..., :2, :]
            # temp_poses = temp_poses.reshape(*temp_poses.shape[:-2], 6)
            # transformed_templates = self.transform_templates(templates, temp_poses)
            # plot_image_tensor((transformed_templates.T * presences.T).T.max(dim=1)[0])

            # Tensor of shape (batch_size, self._n_caps, 6)
            object_poses = self.rand_poses((MNISTObjects.NUM_SAMPLES, 1), size_ratio=6)
            object_poses = math_utils.geometric_transform(object_poses, similarity=True, inverse=True, as_matrix=True)

            jitter_poses = self.rand_jitter_poses((MNISTObjects.NUM_SAMPLES, self.num_caps))
            jitter_poses = math_utils.geometric_transform(jitter_poses, similarity=True, inverse=True, as_matrix=True)

            poses = jitter_poses\
                    @ part_poses.repeat((MNISTObjects.NUM_SAMPLES // MNISTObjects.NUM_CLASSES, 1, 1, 1))\
                    @ object_poses.expand((MNISTObjects.NUM_SAMPLES, self.num_caps, -1, -1))
            poses = poses[..., :2, :]
            poses = poses.reshape(*poses.shape[:-2], 6)

            presences = presences.repeat((MNISTObjects.NUM_SAMPLES // MNISTObjects.NUM_CLASSES, 1))

            if self.template_mixing == 'pdf':
                rec = pcae_decoder(poses, presences)
                images = rec.pdf.mean()
            elif self.template_mixing == 'max':
                transformed_templates = self.transform_templates(self.templates, poses)
                # templates = templates.repeat((MNISTObjects.NUM_SAMPLES // MNISTObjects.NUM_CLASSES, 1))
                images = (transformed_templates.T * presences.T).T.max(dim=1)[0]
            else:
                raise ValueError(f'Invalid template_mixing value {self.template_mixing}')

            self.data = EasyDict(
                images=images,
                templates=pcae_decoder.templates,
                jitter_poses=jitter_poses,
                caps_poses=part_poses,
                sample_poses=object_poses
            )

    def overlap_metric(self, transformed_templates, presences):
        # Tensor of size (N_CLASSES, N_CAPS, C, H, W), transposes are for elem-wise mult broadcasting
        t = (transformed_templates.T * presences.T).T
        # Tensor of size (N_CLASSES, N_CAPS, C*H*W)
        t = t.view(*t.shape[:-3], -1)
        # Tensor of size (N_CLASSES, N_CAPS, N_CAPS, C*H*W)
        metric_t = t.unsqueeze(1) * t.unsqueeze(2)
        # Tensor of size (N_CLASSES, N_CAPS, N_CAPS)
        metric_t = metric_t.sum(dim=-1)
        # Tensor of size (N_CLASSES, N_CAPS, N_CAPS) w/ diag zeroed
        return metric_t * (torch.ones_like(metric_t) - torch.eye(self.num_caps).cuda())

    def rand_poses(self, shape, size_ratio):
        trans_xs = (torch.rand(shape).cuda() - .5) * 1
        trans_ys = (torch.rand(shape).cuda() - .5) * 1
        if self.aligned:
            scale_xs = (torch.rand(shape).cuda() * .9 + .1) * size_ratio
            scale_ys = (torch.rand(shape).cuda() * .9 + .1) * size_ratio
            thetas = (torch.rand(shape).cuda() - .5) * 3.1415 * (6 / 180)
        else:
            scale_xs = torch.rand(shape).cuda() * size_ratio * .9 + .1
            scale_ys = torch.rand(shape).cuda() * size_ratio * .9 + .1
            thetas = torch.rand(shape).cuda() * 2 * 3.1415
        shears = torch.zeros(shape).cuda()
        poses = torch.stack([trans_xs, trans_ys, scale_xs, scale_ys, thetas, shears], dim=-1)
        return poses

    def rand_jitter_poses(self, shape):
        trans_xs = (torch.rand(shape).cuda() - .5) * .1
        trans_ys = (torch.rand(shape).cuda() - .5) * .1
        if self.aligned:
            scale_xs = torch.rand(shape).cuda() * .3 + .85
            scale_ys = torch.rand(shape).cuda() * .3 + .85
            thetas = (torch.rand(shape).cuda() - .5) * 3.1415 * (9 / 180)
        else:
            scale_xs = torch.rand(shape).cuda() * .2 + .9
            scale_ys = torch.rand(shape).cuda() * .2 + .9
            thetas = torch.rand(shape).cuda() * 2 * 3.1415 / 60
        shears = torch.zeros(shape).cuda()
        poses = torch.stack([trans_xs, trans_ys, scale_xs, scale_ys, thetas, shears], dim=-1)
        return poses

    def plot(self, n=10):
        images = torch.stack([self[i][0] for i in range(n)])
        plot_image_tensor(images)

    def __getitem__(self, item):
        """
        Randomly inserts the MNIST images into cifar images
        :param item:
        :return:
        """
        if self.train:
            idx = item
        else:
            idx = MNISTObjects.NUM_SAMPLES * 4 // 5 + item
        image = self.data.images[idx]

        image = self.norm_img(image)

        return image, torch.zeros(1, dtype=torch.long)

    @staticmethod
    def norm_img(image):
        image = torch.abs(image - image.quantile(.5))
        i_max = torch.max(image)
        i_min = torch.min(image)
        image = torch.div(image - i_min, i_max - i_min + 1e-8)
        return image

    def transform_templates(self, templates, poses, output_shape=(40, 40)):
        batch_size = poses.shape[0]
        template_batch_shape = (batch_size, self.num_caps, templates.shape[-3], *output_shape)
        flattened_template_batch_shape = (template_batch_shape[0]*template_batch_shape[1], *template_batch_shape[2:])

        # poses shape (MNISTObjects.NUM_CLASSES * self._n_caps, 2, 3)
        poses = poses.view(-1, 2, 3)
        # TODO: port to using https://kornia.readthedocs.io/en/latest/geometry.transform.html#kornia.geometry.transform.warp_affine
        grid_coords = nn.functional.affine_grid(theta=poses, size=flattened_template_batch_shape)
        template_stack = templates.repeat(batch_size, 1, 1, 1)
        transformed_templates = nn.functional.grid_sample(template_stack, grid_coords).view(template_batch_shape)
        return transformed_templates

    def __len__(self):
        if self.train:
            return MNISTObjects.NUM_SAMPLES * 4 // 5
        return MNISTObjects.NUM_SAMPLES // 5

if __name__ == '__main__':
    # 8 Temp: mlatberkeley/StackedCapsuleAutoEncoders/fm9q1zxd
    # 4 Temp: mlatberkeley/StackedCapsuleAutoEncoders/67lzaiyq
    # MNISTObjects(template_src='mlatberkeley/StackedCapsuleAutoEncoders/fm9q1zxd', num_caps=8)
    ds = MNISTObjects(template_src='mlatberkeley/StackedCapsuleAutoEncoders/67lzaiyq', num_caps=4, new=True)
