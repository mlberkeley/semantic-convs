import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib as pth
from torchvision.datasets import CIFAR10, MNIST, QMNIST, USPS
from torchvision import transforms
import torch
from easydict import EasyDict

from scae.models.pcae import PCAE
from scae.modules.part_capsule_ae import TemplateImageDecoder
from scae.args import parse_args
import scae.util.math as math_utils

class MNISTObjects(torch.utils.data.Dataset):
    NUM_SAMPLES = 10000
    NUM_CLASSES = 10

    def __init__(self, root='data', train=True,
                 template_run_path='mlatberkeley/StackedCapsuleAutoEncoders/67lzaiyq',
                 num_caps=4, new=False):
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
        self._generate(template_run_path)

    def _generate(self, template_run_path):
        with torch.no_grad():
            args = parse_args(f'--cfg scae/config/mnist.yaml --debug'.split(' '))
            args.pcae.num_caps = self.num_caps
            args.im_channels = 1

            pcae_decoder = TemplateImageDecoder(args).cuda()
            if template_run_path is not None:
                from util.wandbapi import download_model
                pcae_decoder.templates = torch.nn.Parameter(download_model(template_run_path)['state_dict']['decoder.templates'].contiguous())


            # Tensor of shape (batch_size, self._n_caps, 6)
            sample_poses = self.rand_poses((MNISTObjects.NUM_SAMPLES, 1),
                                           size_ratio=6)
            sample_poses = math_utils.geometric_transform(sample_poses, similarity=True, inverse=True, as_matrix=True)

            caps_poses = self.rand_poses((MNISTObjects.NUM_CLASSES, self.num_caps),
                                         size_ratio=args.pcae.decoder.template_size[0] / args.pcae.decoder.output_size[0] / 2)
            caps_poses = math_utils.geometric_transform(caps_poses, similarity=True, inverse=True, as_matrix=True)

            jitter_poses = self.rand_jitter_poses((MNISTObjects.NUM_SAMPLES, self.num_caps))
            jitter_poses = math_utils.geometric_transform(jitter_poses, similarity=True, inverse=True, as_matrix=True)

            poses = jitter_poses\
                    @ caps_poses.repeat((MNISTObjects.NUM_SAMPLES // MNISTObjects.NUM_CLASSES, 1, 1, 1))\
                    @ sample_poses.expand((MNISTObjects.NUM_SAMPLES, self.num_caps, -1, -1))
            poses = poses[..., :2, :]
            poses = poses.reshape(*poses.shape[:-2], 6)

            presences_shape = (MNISTObjects.NUM_SAMPLES, self.num_caps)
            presences = torch.sigmoid((torch.rand(presences_shape).cuda() - .5) * 1)

            rec = pcae_decoder(poses, presences)
            rec_img = rec.pdf.mean()
            self.data = EasyDict(
                images=rec_img,
                templates=pcae_decoder.templates,
                jitter_poses=jitter_poses,
                caps_poses=caps_poses,
                sample_poses=sample_poses
            )
            pass

    def rand_poses(self, shape, size_ratio):
        trans_xs = (torch.rand(shape).cuda() - .5) * 1
        trans_ys = (torch.rand(shape).cuda() - .5) * 1
        scale_xs = torch.rand(shape).cuda() * size_ratio * .9 + .1
        scale_ys = torch.rand(shape).cuda() * size_ratio * .9 + .1
        thetas = torch.rand(shape).cuda() * 2 * 3.1415
        shears = torch.zeros(shape).cuda()
        poses = torch.stack([trans_xs, trans_ys, scale_xs, scale_ys, thetas, shears], dim=-1)
        return poses

    def rand_jitter_poses(self, shape):
        trans_xs = (torch.rand(shape).cuda() - .5) * .1
        trans_ys = (torch.rand(shape).cuda() - .5) * .1
        scale_xs = torch.rand(shape).cuda() * .2 + .9
        scale_ys = torch.rand(shape).cuda() * .2 + .9
        thetas = torch.rand(shape).cuda() * 2 * 3.1415 / 60
        shears = torch.zeros(shape).cuda()
        poses = torch.stack([trans_xs, trans_ys, scale_xs, scale_ys, thetas, shears], dim=-1)
        return poses

    def plot(self, n=10):
        for i in range(n):
            plt.imshow(self[i, 0][0].detach().cpu())
            plt.show()

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

        image = torch.abs(image - image.quantile(.5))
        i_max = torch.max(image)
        i_min = torch.min(image)
        image = torch.div(image - i_min, i_max - i_min + 1e-8)

        return image, torch.zeros(1, dtype=torch.long)

    def __len__(self):
        if self.train:
            return MNISTObjects.NUM_SAMPLES * 4 // 5
        return MNISTObjects.NUM_SAMPLES // 5

if __name__ == '__main__':
    # 8 Temp: mlatberkeley/StackedCapsuleAutoEncoders/fm9q1zxd
    # 4 Temp: mlatberkeley/StackedCapsuleAutoEncoders/67lzaiyq
    # MNISTObjects(template_src='mlatberkeley/StackedCapsuleAutoEncoders/fm9q1zxd', num_caps=8)
    ds = MNISTObjects(template_run_path='mlatberkeley/StackedCapsuleAutoEncoders/67lzaiyq', num_caps=4, new=True)
