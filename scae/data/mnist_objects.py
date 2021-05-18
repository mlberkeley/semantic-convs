import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib as pth

import wandb
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
from scae.util.wandb import to_wandb_im

class MNISTObjects(torch.utils.data.Dataset):
    NUM_CLASSES = 10

    def __init__(self, root='data', train=True,
                 template_src='mlatberkeley/StackedCapsuleAutoEncoders/67lzaiyq',
                 num_caps=4, new=True, aligned=True, template_mixing='pdf', size=10000):
        self.train = train
        self.num_caps = num_caps
        self.file = pth.Path(root) / 'mnist_objects.pkl'
        self.size = size
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

            def generate_valid_part_object_relations(num_objects=MNISTObjects.NUM_CLASSES, num_capsules=self.num_caps):
                valid_part_poses = []
                valid_presences = []
                while len(valid_part_poses) < num_objects:
                    presences_shape = (num_objects, self.num_caps)
                    presences = Bernoulli(.99).sample(presences_shape).float().cuda()

                    part_poses = self.rand_poses((num_objects, num_capsules),
                                                 size_ratio=args.pcae.decoder.template_size[0] / args.pcae.decoder.output_size[0] / 2)
                    part_poses = math_utils.geometric_transform(part_poses, similarity=True, inverse=True, as_matrix=True)

                    transformed_templates = self.transform_templates(self.templates, part_poses)
                    metric = self.overlap_metric(transformed_templates, presences)
                    metric = metric * (presences.bool().unsqueeze(-1) | presences.bool().unsqueeze(-2)).float()

                    for i in range(MNISTObjects.NUM_CLASSES):
                        if ((metric[i] == 0) | ((10 < metric[i]) & (metric[i] < 20))).all()\
                                and (metric[i] > 0).any():
                            valid_part_poses.append(part_poses[i])
                            valid_presences.append(presences[i])

                valid_part_poses = torch.stack(valid_part_poses[:MNISTObjects.NUM_CLASSES])
                valid_presences = torch.stack(valid_presences[:MNISTObjects.NUM_CLASSES])
                return valid_part_poses, valid_presences

            self.part_poses, self.presences = generate_valid_part_object_relations()

            # Vis final objects
            transformed_templates = self.transform_templates(self.templates, self.part_poses)
            self.objects = (transformed_templates.T * self.presences.T).T

            # Tensor of shape (batch_size, self._n_caps, 6)
            object_poses = self.rand_poses((self.size, 1), size_ratio=6)
            self.object_poses = math_utils.geometric_transform(object_poses, similarity=True, inverse=True, as_matrix=True)

            jitter_poses = self.rand_jitter_poses((self.size, self.num_caps))
            self.jitter_poses = math_utils.geometric_transform(jitter_poses, similarity=True, inverse=True, as_matrix=True)

            self.poses = self.jitter_poses\
                         @ self.part_poses.repeat((self.size // MNISTObjects.NUM_CLASSES, 1, 1, 1))\
                         @ self.object_poses.expand((self.size, self.num_caps, -1, -1))

            presences = self.presences.repeat((self.size // MNISTObjects.NUM_CLASSES, 1))

            labels = torch.arange(0, MNISTObjects.NUM_CLASSES, step=1, dtype=torch.long)
            self.labels = labels.expand((self.size // MNISTObjects.NUM_CLASSES, -1)).reshape(-1)

            if self.template_mixing == 'pdf':
                rec = pcae_decoder(self.poses[..., :2, :].reshape(*self.poses.shape[:-2], 6), self.presences)
                self.images = rec.pdf.mean()
            elif self.template_mixing == 'max':
                transformed_templates = self.transform_templates(self.templates, self.poses)
                # templates = templates.repeat((self.size // MNISTObjects.NUM_CLASSES, 1))
                self.images = (transformed_templates.T * presences.T).T.max(dim=1)[0]
            else:
                raise ValueError(f'Invalid template_mixing value {self.template_mixing}')

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

    def __getitem__(self, item):
        """
        Randomly inserts the MNIST images into cifar images
        :param item:
        :return:
        """
        idx = item if self.train else self.size * 4 // 5 + item

        image = self.images[idx]
        image = self.norm_img(image)
        return image, self.labels[idx]

    @staticmethod
    def norm_img(image):
        image = torch.abs(image - image.quantile(.5))
        i_max = torch.max(image)
        i_min = torch.min(image)
        image = torch.div(image - i_min, i_max - i_min + 1e-8)
        return image

    def transform_templates(self, templates, poses, output_shape=(40, 40)):
        """

        :param templates:
        :param poses: [MNISTObjects.NUM_CLASSES * self._n_caps, 3, 3] tensor
        :param output_shape:
        :return:
        """
        batch_size = poses.shape[0]
        template_batch_shape = (batch_size, self.num_caps, templates.shape[-3], *output_shape)
        flattened_template_batch_shape = (template_batch_shape[0]*template_batch_shape[1], *template_batch_shape[2:])

        # poses shape
        poses = poses[..., :2, :].reshape(-1, 2, 3)
        # TODO: port to using https://kornia.readthedocs.io/en/latest/geometry.transform.html#kornia.geometry.transform.warp_affine
        grid_coords = nn.functional.affine_grid(theta=poses, size=flattened_template_batch_shape)
        template_stack = templates.repeat(batch_size, 1, 1, 1)
        transformed_templates = nn.functional.grid_sample(template_stack, grid_coords).view(template_batch_shape)
        return transformed_templates

    def __len__(self):
        if self.train:
            return self.size * 4 // 5
        return self.size // 5

    # Vis util functions
    def plot(self, n=NUM_CLASSES):
        objects = self.objects.max(dim=1)[0]
        plot_image_tensor(objects)
        images = torch.stack([self[i][0] for i in range(n)])
        plot_image_tensor(images)

    def log_to_wandb_run(self, run, name="mnist_objects", **artifact_kwargs):
        artifact = wandb.Artifact(name, "dataset", **artifact_kwargs)

        def get_row(idx):
            data = self[idx]
            return [idx, to_wandb_im(data[0]), data[1].item()]

        table = wandb.Table(
            columns=["idx", "image", "label"],
            data=[get_row(idx) for idx in range(len(self))]
        )
        run.log({name: table})

        artifact.add(table, "dataset_table")
        run.log_artifact(artifact)
        pass

if __name__ == '__main__':
    # 8 Temp: mlatberkeley/StackedCapsuleAutoEncoders/fm9q1zxd
    # 4 Temp: mlatberkeley/StackedCapsuleAutoEncoders/67lzaiyq
    # MNISTObjects(template_src='mlatberkeley/StackedCapsuleAutoEncoders/fm9q1zxd', num_caps=8)
    ds = MNISTObjects(template_mixing='max', template_src='mlatberkeley/StackedCapsuleAutoEncoders/67lzaiyq', num_caps=4, new=True)
    ds.plot(100)
