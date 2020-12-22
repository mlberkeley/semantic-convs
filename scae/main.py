import argparse
import os

import wandb
from easydict import EasyDict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from args import parse_args

def main():
    args = parse_args()

    if args.debug or not args.non_deterministic:
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # torch.set_deterministic(True) # grid_sampler_2d_backward_cuda does not have a deterministic implementation

    if args.debug:
        torch.autograd.set_detect_anomaly(True)


    if args.dataset == 'mnist':
        args.num_classes = 10
        args.im_channels = 1

        from torchvision.datasets import MNIST

        t = transforms.Compose([
            transforms.RandomCrop(size=(40, 40), pad_if_needed=True),
            transforms.ToTensor()
        ])

        train_dataset = MNIST('data', train=True, transform=t, download=True)
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.data_workers)

        val_dataset = MNIST('data', train=False, transform=t, download=True)
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.data_workers)
    else:
        raise NotImplementedError()


    logger = WandbLogger(
        project=args.log_project,
        name=args.log_run_name,
        config=args, offline=not args.log_upload)


    if args.model == 'ccae':
        from modules.constellation_ae import SetTransformer, ConstellationCapsule
        from models.ccae import CCAE

        encoder = SetTransformer()
        decoder = ConstellationCapsule()
        model = CCAE(encoder, decoder, args)

        # logger.watch(encoder._encoder, log='all', log_freq=args.log_frequency)
        # logger.watch(decoder, log='all', log_freq=args.log_frequency)
    elif args.model == 'pcae':
        from modules.part_capsule_ae import CapsuleImageEncoder, TemplateImageDecoder
        from models.pcae import PCAE

        encoder = CapsuleImageEncoder(
            args.pcae_num_caps, args.pcae_caps_dim, args.pcae_feat_dim)
        decoder = TemplateImageDecoder(
            args.pcae_num_caps, use_alpha_channel=args.alpha_channel, output_size=(40, 40))
        model = PCAE(encoder, decoder, args)

        logger.watch(encoder._encoder, log='all', log_freq=args.log_frequency)
        logger.watch(decoder, log='all', log_freq=args.log_frequency)
    
    elif args.model == 'ocae':
        from modules.object_capsule_ae import ImageCapsule
        from modules.attention import SetTransformer
        from models.ocae import OCAE
        from modules.part_capsule_ae import CapsuleImageEncoder, TemplateImageDecoder
        from models.pcae import PCAE

        p_encoder = CapsuleImageEncoder(
            args.pcae_num_caps, args.pcae_caps_dim, args.pcae_feat_dim)
        p_decoder = TemplateImageDecoder(
            args.pcae_num_caps, use_alpha_channel=args.alpha_channel, output_size=(40, 40))
        
        encoder = SetTransformer(n_layers=3, n_heads=1, n_dims=16,
                                 n_output_dims=256, n_outputs=10)
        decoder = ImageCapsule(n_caps=10, n_caps_dims=2, n_votes=16,
                               n_caps_params=32, n_hiddens=128, 
                               learn_vote_scale=True, deformations=True,
                               noise_type='uniform', noise_scale=4.,
                               similarity_transform=False)
        model = OCAE(encoder, decoder, p_encoder, p_decoder, args)
        #  TODO: after ccae #
    else:
        raise NotImplementedError()

    # Execute Experiment
    trainer = pl.Trainer(gpus=0, max_epochs=args.num_epochs, logger=logger)
    trainer.fit(model, train_dataloader)

if __name__ == "__main__":
    main()
