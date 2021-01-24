import wandb
from easydict import EasyDict
import torch_optimizer as optim

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


def to_wandb_im(x, **kwargs):  # TODO: move to utils
    x = x.detach()

    if len(x.shape) == 3:
        # Torch uses C, H, W
        x = x.permute(1, 2, 0)

    if x.shape[-1] == 2:
        # channels = val, alpha
        val = x[..., 0]
        alpha = x[..., 1]

        # convert to RGBA
        x = torch.stack([val]*3 + [alpha], dim=-1)

    return wandb.Image(x.cpu().numpy(), **kwargs)

def rec_to_wandb_im(x, **kwargs):  # TODO: move to utils
    # TODO: unpack reconstruction template components
    return to_wandb_im(x, **kwargs)


class PCAE(pl.LightningModule):
    def __init__(self, encoder, decoder, args: EasyDict):
        super(PCAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.lr = args.pcae.lr
        self.lr_decay = args.pcae.lr_decay
        self.weight_decay = args.pcae.weight_decay

        self.n_classes = args.num_classes
        self.mse = nn.MSELoss()

        self.args = args

        # TODO: get checkpointing working
        # self.save_hyperparameters('encoder', 'decoder', 'n_classes', 'args')

    def forward(self, img, labels=None, log=False, log_imgs=False):
        batch_size = img.shape[0]

        # Computation:
        capsules = self.encoder(img)

        rec = self.decoder(capsules.poses, capsules.presences)
        rec_img = rec.pdf.mean()

        # Loss Calculations:
        rec_ll = rec.pdf.log_prob(img).view(batch_size, -1).sum(dim=-1).mean()
        temp_l1 = F.relu(self.decoder.templates).mean()
        rec_mse = self.mse(rec_img, img)

        losses = EasyDict(
            rec_ll=rec_ll.detach(),
            temp_l1=temp_l1.detach(),
            rec_mse=rec_mse.detach()
        )
        losses_scaled = EasyDict(
            rec_ll=-rec_ll * self.args.pcae.loss_ll_coeff,
            temp_l1=temp_l1 * self.args.pcae.loss_temp_l1_coeff,
            rec_mse=rec_mse * self.args.pcae.loss_mse_coeff
        )
        loss = sum([l for l in losses_scaled.values()])

        # Logging:
        if log:
            for k in losses:
                self.log(f'{log}_{k}', losses[k])
            # TODO: replace logging of this with grad-magnitude logging
            #   to understand contribution of each loss independently
            # for k in losses_scaled:
            #     self.log(f'{log}_{k}_scaled', losses[k].detach())
            self.log('epoch', self.current_epoch)

            # TODO: log caps presences
            # self.logger.experiment.log({'capsule_presence': capsules.presences.detach().cpu()}, commit=False)
            # self.logger.experiment.log({'capsule_presence_thres': (capsules.presences > .1).sum(dim=-1)}, commit=False)
            if log_imgs:
                n = 8
                gt_imgs = [to_wandb_im(img[i], caption='gt_image') for i in range(n)]
                rec_imgs = [rec_to_wandb_im(rec_img[i], caption='rec_image') for i in range(n)]
                gt_rec_imgs = [None] * (2 * n)
                gt_rec_imgs[::2], gt_rec_imgs[1::2] = gt_imgs, rec_imgs  # interweave

                template_imgs = [to_wandb_im(t, caption=f'tmp_{i}') for i, t in enumerate(rec.raw_templates)]
                mixture_mean_imgs = [to_wandb_im(t, caption=f'tmp_{i}') for i, t in enumerate(rec.mixture_means[0])]
                mixture_logit_imgs = [to_wandb_im(t, caption=f'tmp_{i}') for i, t in enumerate(rec.mixture_logits[0])]

                # TODO: proper train / val prefixing
                self.logger.experiment.log({
                    'imgs': gt_rec_imgs,
                    'templates': template_imgs,
                    'mixture_means': mixture_mean_imgs,
                    'mixture_logits': mixture_logit_imgs
                }, commit=False)

        return EasyDict(
            loss=loss,
            capsules=capsules,
            reconstruction=rec
        )

    def training_step(self, batch, batch_idx):
        # img    shape (batch_size, C, H, W)
        # labels shape (batch_size)
        img, labels = batch
        ret = self(img, labels, log='train', log_imgs=(batch_idx == 0))

        if torch.isnan(ret.loss).any():  # TODO: try grad clipping?
            raise ValueError('loss is nan')

        return ret.loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        # TODO: add val sets list
        val_set = '' if dataloader_idx is None else f'_{self.val_sets[dataloader_idx]}'
        with torch.no_grad():
            img, labels = batch
            ret = self(img, labels, log=f'val{val_set}', log_imgs=(batch_idx % 20 == 0))
        return ret.loss

    def configure_optimizers(self):
        param_sets = [
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters(), 'lr': self.lr * self.args.pcae.decoder.lr_coeff}
        ]
        if self.args.pcae.optimizer == 'sgd':
            opt = torch.optim.SGD(param_sets, lr=self.lr, weight_decay=self.weight_decay)
        elif self.args.pcae.optimizer == 'radam':
            opt = optim.RAdam(param_sets, lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError()

        if self.args.pcae.lr_scheduler == 'exp':
            scheduler_step = 'epoch'
            lr_sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.lr_decay)
        elif self.args.pcae.lr_scheduler == 'cosrestarts':
            scheduler_step = 'step'
            lr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 469*8)  # TODO scale by batch num
        else:
            raise NotImplementedError

        return [opt], [{
            'scheduler': lr_sched,
            'interval': scheduler_step,
            'name': 'pcae'
        }]
