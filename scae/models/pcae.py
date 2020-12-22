import wandb
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as O

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
    def __init__(self, encoder, decoder, args):
        super(PCAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.lr = args.pcae_lr
        self.lr_decay = args.pcae_lr_decay
        self.weight_decay = args.pcae_weight_decay

        self.n_classes = args.num_classes
        self.mse = nn.MSELoss()

        self.args = args

    def forward(self, image):
        capsules = self.encoder(image)

        presences = capsules.presences.detach().cpu()
        self.logger.experiment.log({'capsule_presence': presences}, commit=False)
        self.logger.experiment.log({'capsule_presence_thres': (presences > .1).sum(dim=-1)}, commit=False)

        reconstruction = self.decoder(capsules.poses, capsules.presences)
        return capsules, reconstruction

    def training_step(self, batch, batch_idx):
        # img    shape (batch_size, C, H, W)
        # labels shape (batch_size)
        img, labels = batch
        batch_size = img.shape[0]

        capsules, rec = self(img)
        rec_ll = rec.pdf.log_prob(img).view(batch_size, -1).sum(dim=-1).mean()
        self.log('rec_log_likelihood', rec_ll, prog_bar=True)

        temp_l1 = F.relu(self.decoder.templates).mean()
        self.log('temp_l1', temp_l1, prog_bar=True)

        rec_mse = self.mse(rec.pdf.mean(), img)
        self.log('rec_mse', rec_mse.detach(), prog_bar=True)

        if batch_idx == 10: #% 10 == 0:
            n = 8
            gt_imgs = [to_wandb_im(img[i], caption='gt_image') for i in range(n)]
            rec_imgs = [rec_to_wandb_im(rec.pdf.mean(idx=i), caption='rec_image') for i in range(n)]
            gt_rec_imgs = [None]*(2*n)
            gt_rec_imgs[::2], gt_rec_imgs[1::2] = gt_imgs, rec_imgs  # interweave

            template_imgs = [to_wandb_im(t, caption=f'tmp_{i}') for i, t in enumerate(rec.raw_templates)]
            mixture_mean_imgs = [to_wandb_im(t, caption=f'tmp_{i}') for i, t in enumerate(rec.mixture_means[0])]
            mixture_logit_imgs = [to_wandb_im(t, caption=f'tmp_{i}') for i, t in enumerate(rec.mixture_logits[0])]

            self.logger.experiment.log({
                'train_imgs': gt_rec_imgs,
                'templates': template_imgs,
                'mixture_means': mixture_mean_imgs,
                'mixture_logits': mixture_logit_imgs,
                'epoch': self.current_epoch},
                commit=False)

        loss = -rec_ll * self.args.pcae_loss_ll_coeff + \
               temp_l1 * self.args.pcae_loss_temp_l1_coeff + \
               rec_mse * self.args.pcae_loss_mse_coeff

        if torch.isnan(loss).any():  # TODO: try grad clipping?
            raise ValueError('loss is nan')

        return loss

    def training_epoch_end(self, outputs):
        ...
        # self.log('train_acc_epoch', self.accuracy.compute())

    def configure_optimizers(self):
        if self.args.pcae_optimizer == 'sgd':
            opt = torch.optim.SGD([
                {'params': self.encoder.parameters()},
                {'params': self.decoder.parameters(), 'lr': self.lr * self.args.pcae_decoder_lr_coeff}
            ], lr=self.lr, weight_decay=self.weight_decay)
        elif self.args.pcae_optimizer == 'radam':
            opt = O.RAdam([
                {'params': self.encoder.parameters()},
                {'params': self.decoder.parameters(), 'lr': self.lr * self.args.pcae_decoder_lr_coeff}
            ], lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError()

        if self.args.pcae_lr_scheduler == 'exp':
            scheduler_step = 'epoch'
            lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=self.lr_decay)
        elif self.args.pcae_lr_scheduler == 'cosrestarts':
            scheduler_step = 'step'
            lr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 469*8)
        else:
            raise NotImplementedError

        return [opt], [{
            'scheduler': lr_sched,
            'interval': scheduler_step,
            'name': 'pcae'
        }]
