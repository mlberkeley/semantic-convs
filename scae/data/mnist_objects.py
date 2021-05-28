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
                 num_caps=4, new=True, aligned=True, template_mixing='sum', size=10000):
        self.train = train
        self.num_caps = num_caps
        self.file = pth.Path(root) / 'mnist_objects.pkl'
        self.size = size
        # TODO: implement data saving and loading (via wandb artifacts or shared random-seed)
        # if self.file.exists() and not new:
        #     with open(self.file, 'rb') as f:
        #         self.data = pickle.load(f)
        # else:
        #     with open(self.file, 'wb') as f:
        #         self._generate(template_src)
        #         pickle.dump(self.data, f)

        self.aligned = aligned
        self.template_mixing = template_mixing

        self._init_templates(template_src)
        self.object_size = (28, 28)
        self.output_size = (40, 40)

        self._generate_objects(self.num_caps)
        self._generate_dataset()

    def _init_templates(self, template_src):
        args = parse_args(f'--cfg scae/config/mnist.yaml --debug'.split(' '))
        args.pcae.num_caps = self.num_caps
        args.im_channels = 1

        pcae_decoder = TemplateImageDecoder(args).cuda()
        if template_src is not None:
            import wandb
            from pytorch_lightning.utilities.cloud_io import load as pl_load
            best_model = wandb.restore('last.ckpt', run_path=template_src, replace=True)
            pcae_decoder.templates = torch.nn.Parameter(
                pl_load(best_model.name)['state_dict']['decoder.templates'].contiguous(), requires_grad=False
            )
        else:
            return NotImplementedError("Dataset templates need to be fetched from wandb run")

        self.__pcae_decoder = pcae_decoder
        self.templates = pcae_decoder._template_nonlin(pcae_decoder.templates)
        self.part_size = args.pcae.decoder.template_size

    def _generate_objects(self, num_capsules, num_objects=NUM_CLASSES):
        valid_part_poses = []
        valid_part_presences = []
        # valid_overlaps = []
        while len(valid_part_poses) < num_objects:
            part_presences_shape = (num_objects, self.num_caps)
            # presences = Bernoulli(.99).sample(presences_shape).float().cuda()
            part_presences = torch.ones(part_presences_shape).cuda()

            part_poses = self._rand_part_to_obj_poses(
                (num_objects, num_capsules),
                size_ratio=self.part_size[0] / self.object_size[0]
            )
            part_poses = math_utils.geometric_transform(part_poses, similarity=True, inverse=True, as_matrix=True)

            transformed_templates = self._transform_templates(self.templates, part_poses)
            valid, overlaps = self._overlap_condition(transformed_templates, part_presences)

            for i in range(MNISTObjects.NUM_CLASSES):
                if valid[i]:
                    valid_part_poses.append(part_poses[i])
                    valid_part_presences.append(part_presences[i])
                    # valid_overlaps.append(overlaps[i])

        self.part_obj_poses = torch.stack(valid_part_poses[:MNISTObjects.NUM_CLASSES])
        self.part_presences = torch.stack(valid_part_presences[:MNISTObjects.NUM_CLASSES])

        self.objects = self._mix_templates(self.templates, self.part_obj_poses, self.part_presences)

    def _generate_dataset(self):
        with torch.no_grad():
            jitter_poses = self._rand_part_jitter_poses((self.size, self.num_caps))
            self.jitter_poses = math_utils.geometric_transform(jitter_poses, similarity=False, inverse=True,
                                                               as_matrix=True)

            # Tensor of shape (batch_size, 1, 6)
            obj_scene_poses = self._rand_obj_to_scene_poses((self.size, 1), size_ratio=self.object_size[0] / self.output_size[0], aligned=self.aligned)
            self.obj_scene_poses = math_utils.geometric_transform(obj_scene_poses, similarity=True, inverse=True, as_matrix=True)

            # Tensor of shape (dataset size, self._n_caps, 6)
            self.part_scene_poses = self.jitter_poses\
                                    @ self.part_obj_poses.repeat((self.size // MNISTObjects.NUM_CLASSES, 1, 1, 1))\
                                    @ self.obj_scene_poses.expand((-1, self.num_caps, -1, -1))

            part_presences = self.part_presences.repeat((self.size // MNISTObjects.NUM_CLASSES, 1))

            labels = torch.arange(0, MNISTObjects.NUM_CLASSES, step=1, dtype=torch.long)
            self.labels = labels.expand((self.size // MNISTObjects.NUM_CLASSES, -1)).reshape(-1)

            self.images = self._mix_templates(self.templates, self.part_scene_poses, part_presences)

    def _overlap_condition(self, transformed_templates, presences):
        # TODO: currently assumes presence values are all 1
        connected_thresh = 4
        connected_max = 10

        # Tensor of size (N_CLASSES, N_CAPS, C, H, W), transposes are for elem-wise mult broadcasting
        selected_templates = (transformed_templates.T * presences.T).T
        # selected_templates = [transformed_templates[i, presences[i].type(torch.bool)] for i in
        #                       range(transformed_templates.shape[0])]

        # Tensor of size (N_CLASSES, N_CAPS, C*H*W)
        template_vecs = selected_templates.view(*selected_templates.shape[:-3], -1)
        # Before sum: tensor of size (N_CLASSES, N_CAPS, N_CAPS, C*H*W)
        # After sum:  tensor of size (N_CLASSES, N_CAPS, N_CAPS)
        overlap_matricies = (template_vecs.unsqueeze(1) * template_vecs.unsqueeze(2)).sum(dim=-1)
        overlap_matricies = overlap_matricies * (1 - torch.eye(self.num_caps).cuda())

        adjacency_matricies = (overlap_matricies >= connected_thresh).float()

        # An object is connected iff any row of it's (adjacency matrix) ** num_parts is all positive
        # ** is a matrix power, not element-wise
        connectivity_matricies = adjacency_matricies.matrix_power(transformed_templates.shape[1])
        connected = ((connectivity_matricies >= connected_thresh).sum(-1) == transformed_templates.shape[1]).any(dim=-1)

        # Zero out connections to non-present parts
        # overlap_matricies = overlap_matricies * (presences.bool().unsqueeze(-1) | presences.bool().unsqueeze(-2)).float()

        # Enforce overlap maximum for all overlap matrix values
        overlaps_valid = ((overlap_matricies < connected_thresh) | (overlap_matricies < connected_max)).all(dim=-1).all(dim=-1)

        return connected & overlaps_valid, overlap_matricies

    def _rand_part_to_obj_poses(self, shape, size_ratio):
        trans_xs = (torch.rand(shape).cuda() - .5) * 1
        trans_ys = (torch.rand(shape).cuda() - .5) * 1
        scale = (torch.rand(shape).cuda() * .1 + .95) * size_ratio
        thetas = torch.rand(shape).cuda() * 2 * 3.1415
        z = torch.zeros(shape).cuda()
        poses = torch.stack([trans_xs, trans_ys, scale, z, thetas, z], dim=-1)
        return poses

    def _rand_obj_to_scene_poses(self, shape, size_ratio, aligned):
        trans_xs = (torch.rand(shape).cuda() - .5) * 1
        trans_ys = (torch.rand(shape).cuda() - .5) * 1
        if aligned:
            scale = (torch.rand(shape).cuda() * .1 + .95) * size_ratio
            thetas = (torch.rand(shape).cuda() - .5) * 3.1415 * (20 / 180)
        else:
            scale = torch.rand(shape).cuda() * size_ratio * .9 + .1
            thetas = torch.rand(shape).cuda() * 2 * 3.1415
        z = torch.zeros(shape).cuda()
        poses = torch.stack([trans_xs, trans_ys, scale, z, thetas, z], dim=-1)
        return poses

    def _rand_part_jitter_poses(self, shape):
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

    def _transform_templates(self, templates, poses, output_shape=(40, 40)):
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

    def _mix_templates(self, templates, poses, presences):
        if self.template_mixing == 'pdf':
            rec = self.__pcae_decoder(poses[..., :2, :].reshape(*poses.shape[:-2], 6), presences)
            return rec.pdf.mean()
        else:
            transformed_templates = self._transform_templates(templates, poses)
            transformed_templates = (transformed_templates.T * presences.T).T
            if self.template_mixing == 'max':
                return transformed_templates.max(dim=1)[0]
            elif self.template_mixing == 'sum':
                return transformed_templates.sum(dim=1)
            else:
                raise ValueError(f'Invalid template_mixing value {self.template_mixing}')

    def __getitem__(self, item):
        """
        Randomly inserts the MNIST images into cifar images
        :param item:
        :return:
        """
        idx = item if self.train else self.size * 4 // 5 + item

        image = self._norm_img(self.images[idx])
        return image, self.labels[idx]

    @staticmethod
    def _norm_img(image):
        image = torch.abs(image - image.quantile(.5))
        i_max = torch.max(image)
        i_min = torch.min(image)
        image = torch.div(image - i_min, i_max - i_min + 1e-8)
        return image

    def __len__(self):
        if self.train:
            return self.size * 4 // 5
        return self.size // 5

    # Vis util functions
    def plot(self, n=NUM_CLASSES):
        plot_image_tensor(self._norm_img(self.objects))
        images = torch.stack([self[i][0] for i in range(n)])
        plot_image_tensor(images)

    def log_as_artifact(self, name="mnist_objects", **artifact_kwargs):
        object_table = wandb.Table(
            columns=["object", "label"],
            data=[[self._norm_img(self.objects[idx]), idx] for idx in range(len(self.objects))]
        )

        def get_row(idx):
            data = self[idx]
            return [idx, to_wandb_im(data[0]), data[1].item()]
        dataset_table = wandb.Table(
            columns=["idx", "image", "label"],
            data=[get_row(idx) for idx in range(len(self))]
        )

        artifact = wandb.Artifact(name, "dataset", **artifact_kwargs)
        artifact.add(object_table, "object_table")
        artifact.add(dataset_table, "dataset_table")
        wandb.log_artifact(artifact)

if __name__ == '__main__':
    # 8 Temp: mlatberkeley/StackedCapsuleAutoEncoders/fm9q1zxd
    # 4 Temp: mlatberkeley/StackedCapsuleAutoEncoders/67lzaiyq
    # MNISTObjects(template_src='mlatberkeley/StackedCapsuleAutoEncoders/fm9q1zxd', num_caps=8)
    ds = MNISTObjects(template_mixing='max', size=100, template_src='mlatberkeley/StackedCapsuleAutoEncoders/67lzaiyq', num_caps=4, new=True)
    ds.plot(100)
