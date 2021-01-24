import wandb
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


class OCAE(pl.LightningModule):

    """Reconstruct parts from object predictions.

    TODO: More information from paper here...
    """

    def __init__(self, encoder, decoder, p_encoder, p_decoder, args):
        super(OCAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.p_encoder = p_encoder
        self.p_decoder = p_decoder
        self.lr = args.ocae_lr
        self.lr_decay = args.ocae_lr_decay
        self.weight_decay = args.ocae_weight_decay
        self.n_classes = args.num_classes
        self.mse = nn.MSELoss()

    def forward(self, points):
        """Forward pass.

        Additional details here..."""
        primary_caps = self.p_encoder(points)
        capsules = self.encoder(points)
        reconstruction = self.decoder(capsules.poses, capsules.presences)
        return capsules, reconstruction

    def training_step(self, batch, batch_idx):
        ## TODO ##
        batch = batch.to("cuda")
        primary_caps = self.p_encoder(batch[0])
        pres = primary_caps.presences
        pose = primary_caps.poses
        expanded_pres = pres.unsqueeze(dim=-1)
        input_pose = torch.cat([pose, 1. - expanded_pres], -1)
        input_pose = torch.cat([input_pose, primary_caps.features], -1)
        n_templates = int(primary_caps.poses.shape[1])
        # templates = self.p_decoder(n_templates, primary_caps.features)
        
        
        # input_pose = torch.ones((32, 11, 2))
        
        print("INPUT POSE", input_pose.shape, "INPUT PRES", pres.shape)
        
        h = self.encoder(input_pose)
        
        
        # h = torch.rand(128, 32, 256)
        target_pose, target_pres = pose, pres
        print("H SHAPE", h.shape)
        
        # res = self.decoder(h, target_pose, target_pres)
        """
        img, labels = batch
        batch_size = img.shape[0]
        capsules, rec = self(img)
        rec_ll = rec.pdf.log_prob(img).view(batch_size, -1).sum(dim=-1).mean()
        self.log('rec_log_likelihood', rec_ll, prog_bar=True)

        temp_l1 = F.relu(self.decoder.templates).mean()
        self.log('temp_l1', temp_l1, prog_bar=True)

        rec_mse = self.mse(rec.pdf.mean(), img)
        self.log('rec_mse', rec_mse.detach(), prog_bar=True)
        
        if batch_idx == 100: #% 10 == 0:
            n = 8
            gt_imgs = [to_wandb_im(img[i], caption='gt_image') for i in range(n)]
            rec_imgs = [rec_to_wandb_im(rec.pdf.mean(idx=i), caption='rec_image') for i in range(n)]
            gt_rec_imgs = [None]*(2*n)
            gt_rec_imgs[::2], gt_rec_imgs[1::2] = gt_imgs, rec_imgs  # interweave

            trans_template_imgs = [to_wandb_im(t, caption=f'tmp_{i}') for i, t in enumerate(rec.trans_templates)]
            template_imgs = [to_wandb_im(t, caption=f'tmp_{i}') for i, t in enumerate(rec.raw_templates)]
            mixture_mean_imgs = [to_wandb_im(t, caption=f'tmp_{i}') for i, t in enumerate(rec.mixture_means[0])]
            mixture_logit_imgs = [to_wandb_im(t, caption=f'tmp_{i}') for i, t in enumerate(rec.mixture_logits[0])]

            self.logger.experiment.log({
                'train_imgs': gt_rec_imgs,
                'templates': template_imgs,
                'trans_templates': trans_template_imgs, # todo(maximsmol): typo
                'mixture_means': mixture_mean_imgs,
                'mixture_logits': mixture_logit_imgs,
                'epoch': self.current_epoch},
                commit=False)
        """
        """
        loss = -rec_ll * self.args.pcae_loss_ll_coeff + \
               temp_l1 * self.args.pcae_loss_temp_l1_coeff + \
               rec_mse * self.args.pcae_loss_mse_coeff

        if torch.isnan(loss).any():  # TODO: try grad clipping?
            raise ValueError('loss is nan')

        return loss
        """
    def configure_optimizers(self):
        # TODO
        # Pulled from domas' code
        pass
