import argparse
import os
from collections import defaultdict
from pathlib import Path

import wandb
from easydict import EasyDict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

import torch

from vsa import VSA, ctvec

import res_utils_torch as ru

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning.callbacks as cb

from scae.data.mnist_objects import MNISTObjects

from resonator import VSA, ctvec, Resonator, gen_image_vsa_vecs, gen_pos_dicts, svd_whiten

if __name__ == "__main__":
    dataset = MNISTObjects(train=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    N = int(3e4)
    template_size = (11, 11) # aka template_size
    image_size = (40, 40)

    with torch.no_grad():
        Vt = ctvec(N, image_size[1])
        Ht = ctvec(N, image_size[0])
        vsa = VSA(Vt, Ht, None, image_size, device='cuda')
        template_vsa = VSA(Vt, Ht, None, template_size, device='cuda')

        template_ims = dataset.templates
        template_dict = gen_image_vsa_vecs(template_ims, template_vsa)
        template_ims_w = svd_whiten(template_ims)
        template_dict_w = gen_image_vsa_vecs(template_ims_w, template_vsa)

        attr_dicts = [template_dict_w] + gen_pos_dicts(vsa.Vt, vsa.Ht, image_size)
        res = Resonator(attr_dicts)

        images = []
        for i, batch in enumerate(dataloader):
            image, label = batch
            image = image[0] # batch size 1
            plt.imshow(image.cpu())
            plt.show()
            images.append(image)

            bound_vec = vsa.encode_pix(image)

            res_hist, nsteps = res.decode(bound_vec, 200)

            plt.figure(figsize=(8, 3))
            ru.resplot_im([h.cpu() for h in res_hist], nsteps)  # , labels=res_xlabels, ticks=res_xticks)
            plt.tight_layout()
            plt.show()

            if i > 10:
                break





    pass
