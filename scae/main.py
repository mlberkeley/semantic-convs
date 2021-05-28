import argparse
import os
from pathlib import Path

import wandb
from easydict import EasyDict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning.callbacks as cb

from scae.args import parse_args
from scae.util.filtering import sobel_filter
data_path = Path('data')

norm_3c = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
norm_1c = transforms.Normalize([0.449], [0.226])

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

    dataloader_args = EasyDict(batch_size=args.batch_size, shuffle=False,
                               num_workers=0 if args.debug else args.data_workers)
    if 'mnist' in args.dataset:
        args.num_classes = 10
        args.im_channels = 1
        args.image_size = (40, 40)

        if 'objects' in args.dataset:
            from data.mnist_objects import MNISTObjects

            dataset = MNISTObjects(data_path, train=True)
            train_dataloader = DataLoader(dataset, **dataloader_args)
            val_dataloader = DataLoader(MNISTObjects(data_path, train=False),
                                        **dataloader_args)
        else:
            from torchvision.datasets import MNIST

            t = transforms.Compose([
                transforms.RandomCrop(size=args.pcae.decoder.output_size, pad_if_needed=True),
                transforms.ToTensor(),
                # norm_1c
            ])
            train_dataloader = DataLoader(MNIST(data_path/'mnist', train=True, transform=t, download=True),
                                          **dataloader_args)
            val_dataloader = DataLoader(MNIST(data_path/'mnist', train=False, transform=t, download=True),
                                        **dataloader_args)
    elif 'usps' in args.dataset:
        args.num_classes = 10
        args.im_channels = 1
        args.image_size = (40, 40)

        from torchvision.datasets import USPS

        t = transforms.Compose([
            transforms.RandomCrop(size=args.pcae.decoder.output_size, pad_if_needed=True),
            transforms.ToTensor(),
            # norm_1c
        ])
        train_dataloader = DataLoader(USPS(data_path/'usps', train=True, transform=t, download=True),
                                      **dataloader_args)
        val_dataloader = DataLoader(USPS(data_path/'usps', train=False, transform=t, download=True),
                                    **dataloader_args)
    elif 'cifar10' in args.dataset:
        args.num_classes = 10
        args.im_channels = 3
        args.image_size = (32, 32)

        from torchvision.datasets import CIFAR10

        t = transforms.Compose([
            transforms.RandomCrop(size=args.pcae.decoder.output_size, pad_if_needed=True),
            transforms.ToTensor()
        ])
        train_dataloader = DataLoader(CIFAR10(data_path/'cifar10', train=True, transform=t, download=True),
                                      **dataloader_args)
        val_dataloader = DataLoader(CIFAR10(data_path/'cifar10', train=False, transform=t, download=True),
                                    **dataloader_args)
    elif 'svhn' in args.dataset:
        args.num_classes = 10
        args.im_channels = 1
        args.image_size = (32, 32)

        from torchvision.datasets import SVHN

        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(sobel_filter),
            transforms.ToPILImage(),
            transforms.RandomCrop(size=args.pcae.decoder.output_size, pad_if_needed=True),
            transforms.ToTensor(),
        ])
        train_dataloader = DataLoader(SVHN(data_path/'svhn', split='train', transform=t, download=True),
                                      **dataloader_args)
        val_dataloader = DataLoader(SVHN(data_path/'svhn', split='test', transform=t, download=True),
                                    **dataloader_args)
    else:
        raise NotImplementedError()

    if '{' in args.log.run_name:
        args.log.run_name = args.log.run_name.format(**args)
    logger = WandbLogger(
        project=args.log.project,
        name=args.log.run_name,
        entity=args.log.team,
        config=args, offline=not args.log.upload
    )

    if args.model == 'ccae':
        from scae.modules.constellation_ae import SetTransformer, ConstellationCapsule
        from scae.models.ccae import CCAE

        encoder = SetTransformer()
        decoder = ConstellationCapsule()
        model = CCAE(encoder, decoder, args)

        # logger.watch(encoder._encoder, log='all', log_freq=args.log_frequency)
        # logger.watch(decoder, log='all', log_freq=args.log_frequency)
    elif args.model == 'pcae':
        from scae.modules.part_capsule_ae import CapsuleImageEncoder, TemplateImageDecoder
        from scae.models.pcae import PCAE

        encoder = CapsuleImageEncoder(args)
        decoder = TemplateImageDecoder(args)
        model = PCAE(encoder, decoder, args)

        logger.watch(encoder._encoder, log='all', log_freq=args.log.frequency)
        logger.watch(decoder, log='all', log_freq=args.log.frequency)
    elif args.model == 'ocae':
        from scae.modules.object_capsule_ae import SetTransformer, ImageCapsule
        from scae.models.ocae import OCAE

        encoder = SetTransformer()
        decoder = ImageCapsule()
        model = OCAE(encoder, decoder, args)

        #  TODO: after ccae
    else:
        raise NotImplementedError()

    # dataset.log_to_wandb_run(wandb.run)
    if 'mnist' in args.dataset and 'objects' in args.dataset:
        wandb.log({"dataset_templates": [wandb.Image(i.detach().cpu().numpy(), caption="Label") for i in dataset.data.templates]})
        wandb.log({"dataset_images": [wandb.Image(i.detach().cpu().numpy(), caption="Label") for i in dataset.data.images[:50]]})

    # Execute Experiment
    lr_logger = cb.LearningRateMonitor(logging_interval='step')
    best_checkpointer = cb.ModelCheckpoint(save_top_k=1, monitor='val_rec_ll', filepath=logger.experiment.dir)
    last_checkpointer = cb.ModelCheckpoint(save_last=True, filepath=logger.experiment.dir)
    trainer = pl.Trainer(gpus=1, max_epochs=args.num_epochs, logger=logger,
                         callbacks=[lr_logger, best_checkpointer, last_checkpointer])
    trainer.fit(model, train_dataloader, val_dataloader)
    model.upload_tables()

if __name__ == "__main__":
    main()
