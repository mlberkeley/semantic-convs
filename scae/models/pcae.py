import time
import datetime

import wandb
from easydict import EasyDict
import torch_optimizer as optim

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from scae.util.math import presence_l2_sparsity
from scae.util.wandb import to_wandb_im, rec_to_wandb_im
from scae.util.plots import pdf_grid_plot, pdf_grid_plot_expanded, fig_to_wandb_im


class PCAE(pl.LightningModule):
    def __init__(self, encoder, decoder, args: EasyDict):
        super(PCAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.lr = args.pcae.lr
        self.lr_decay = args.pcae.lr_decay
        self.weight_decay = args.pcae.weight_decay

        self.n_classes = args.num_classes
        self.mse = nn.MSELoss(reduction='none')

        self.args = args

        # TODO: get checkpointing working
        # self.save_hyperparameters('encoder', 'decoder', 'n_classes', 'args')

    def forward(self, imgs, labels=None, log=False, log_imgs=False, batch_idx=None):
        batch_size = imgs.shape[0]

        # Computation:
        capsules = self.encoder(imgs)

        rec = self.decoder(capsules.pose_mats, capsules.presences)
        rec_imgs = rec.pdf.mean()

        # Loss Calculations:
        rec_lls = rec.pdf.log_prob(imgs).view(batch_size, -1).sum(dim=-1)
        rec_ll = rec_lls.mean()
        temp_l1 = F.relu(self.decoder.templates).mean()
        rec_mses = self.mse(rec_imgs, imgs).view(batch_size, -1).mean(dim=-1)
        rec_mse = rec_mses.mean()
        pres_l2_sparsity_over_capsules, pres_l2_sparsity_over_batch =\
            presence_l2_sparsity(capsules.presences, num_classes=self.n_classes)

        losses = EasyDict(
            rec_ll=rec_ll.detach(),
            temp_l1=temp_l1.detach(),
            rec_mse=rec_mse.detach(),
            pres_l2_sparsity_over_capsules=pres_l2_sparsity_over_capsules.detach(),
            pres_l2_sparsity_over_batch=pres_l2_sparsity_over_batch.detach()
        )

        losses_scaled = EasyDict(
            rec_ll=-rec_ll * self.args.pcae.loss_ll_coeff,
            temp_l1=temp_l1 * self.args.pcae.loss_temp_l1_coeff,
            rec_mse=rec_mse * self.args.pcae.loss_mse_coeff,
            pres_l2_sparsity_over_capsules=pres_l2_sparsity_over_capsules * self.args.pcae.loss_pres_l2_sparsity.capsules,
            pres_l2_sparsity_over_batch=pres_l2_sparsity_over_batch * self.args.pcae.loss_pres_l2_sparsity.batch
        )
        loss = sum([l for l in losses_scaled.values()])

        # Logging:
        if log:
            for k in losses:
                self.log(f'{k}/{log}', losses[k])
            # TODO: replace logging of this with grad-magnitude logging
            #   to understand contribution of each loss independently
            # for k in losses_scaled:
            #     self.log(f'{log}_{k}_scaled', losses[k].detach())
            self.log('epoch', self.current_epoch)

            # TODO: log caps presences
            # self.logger.experiment.log({'capsule_presence': capsules.presences.detach().cpu()}, commit=False)
            # self.logger.experiment.log({'capsule_presence_thres': (capsules.presences > .1).sum(dim=-1)}, commit=False)
            if log_imgs and "val" not in log:
                n = 8
                gt_wandb_imgs = [to_wandb_im(imgs[i], caption='gt_image') for i in range(n)]
                rec_wandb_imgs = [rec_to_wandb_im(rec_imgs[i], caption='rec_image') for i in range(n)]
                gt_rec_wandb_imgs = [None] * (2 * n)
                gt_rec_wandb_imgs[::2], gt_rec_wandb_imgs[1::2] = gt_wandb_imgs, rec_wandb_imgs  # interweave

                template_imgs = [to_wandb_im(t, caption=f'tmp_{i}') for i, t in enumerate(rec.raw_templates)]
                mixture_mean_imgs = [to_wandb_im(t, caption=f'tmp_{i}') for i, t in enumerate(rec.mixture_means[0])]
                mixture_logit_imgs = [to_wandb_im(t, caption=f'tmp_{i}') for i, t in enumerate(rec.mixture_logits[0])]

                # TODO: proper train / val prefixing
                self.logger.experiment.log({
                    f'imgs/{log}': gt_rec_wandb_imgs,
                    f'templates/{log}': template_imgs,
                    f'mixture_means/{log}': mixture_mean_imgs,
                    f'mixture_logits/{log}': mixture_logit_imgs
                }, commit=False)

                for idx in range(batch_size):
                    # if idx < 20:
                    #     self.add_row_to_tables("img_table", log, batch_idx, idx, labels, rec_ll, rec_mse, capsules,
                    #                            rec, imgs, rec_imgs)
                    self.add_row_to_tables("scalar_table", log, batch_idx, idx, labels, rec_lls, rec_mses, capsules)

        if self.current_epoch == self.args.num_epochs - 1 and "val" in log:
            for idx in range(batch_size):
                self.add_row_to_tables(f"result_table", log, batch_idx, idx, labels, rec_lls, rec_mses, capsules,
                                       rec, imgs, rec_imgs)

        return EasyDict(
            loss=loss,
            capsules=capsules,
            reconstruction=rec
        )

    def training_step(self, batch, batch_idx):
        # img    shape (batch_size, C, H, W)
        # labels shape (batch_size)
        img, labels = batch
        ret = self(img, labels, log='train', log_imgs=(batch_idx == 0), batch_idx=batch_idx)

        if torch.isnan(ret.loss).any():  # TODO: try grad clipping?
            raise ValueError('loss is nan')
        return ret.loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        val_set = '' if dataloader_idx is None else f'_{self.val_set_names[dataloader_idx]}'
        with torch.no_grad():
            img, labels = batch
            ret = self(img, labels, log=f'val{val_set}', log_imgs=(batch_idx == 0), batch_idx=batch_idx)

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

    def init_tables(self):
        template_images = []
        template_scalars = []
        for i in range(self.args.pcae.num_caps):
            for p in ["transformed_img"]:
                template_images.append(f"c{i}_{p}")
            for p in ["presence",
                      "trans_x", "trans_y", "scale_x", "scale_y", "theta", "shear"]:
                template_scalars.append(f"c{i}_{p}")

        images = ["gt_img", "rec_img"]
        scalars = ["dataset", "mode", "epoch", "batch", "idx", "label", "rec_ll", "mse"]

        # self.img_table = wandb.Table(
        #     columns=[*images, *template_images, *scalars, *template_scalars]
        # )
        self.scalar_table = wandb.Table(
            columns=[*scalars, *template_scalars]
        )
        self.result_table = wandb.Table(
            columns=[*images, *template_images, *scalars, *template_scalars]
        )

    # TODO: add ssim, psnr (kornia)
    def add_row_to_tables(self, table, mode, batch_idx, idx, labels, rec_lls, mses, capsules,
                          rec=None, imgs=None, rec_imgs=None):
        # scalars_only = None not in (rec, imgs, rec_img) TODO: rec is null but still has attributes somehow?
        scalars_only = None in (rec, imgs, rec_imgs)
        if not hasattr(self, table):
            self.init_tables()

        label = labels[idx] if labels is not None else -1

        template_images = []
        template_scalars = []
        for capsule_idx in range(self.args.pcae.num_caps):
            if not scalars_only:
                template_images += [
                    to_wandb_im(rec.mixture_means[idx, capsule_idx]),
                ]
            template_scalars += [
                capsules.presences[idx, capsule_idx].item(),
                *capsules.poses[idx, capsule_idx].detach().cpu().numpy()
            ]
        row = []
        if not scalars_only:
            row += [
                # "gt_img", "rec_img"
                to_wandb_im(imgs[idx]), to_wandb_im(rec_imgs[idx]),
                # templates X "transformed_img"
                *template_images,
            ]
        row += [
            # "dataset", "mode", "epoch", "batch", "idx", "label", "rec_ll", "mse"
            self.args.dataset.name, mode, self.current_epoch, batch_idx, self.args.batch_size * batch_idx + idx,
            label.item(), rec_lls[idx].item(), mses[idx].item(),
            # templates X ["presence", pose]
            *template_scalars
        ]
        getattr(self, table).add_data(*row)

    def vis_tables(self):
        df = pd.DataFrame(data=self.result_table.data, columns=self.result_table.columns)
        for c in range(self.args.pcae.num_caps):
            # map thetas to [-pi, pi]
            df[f"c{c}_theta"] = (df[f"c{c}_theta"] + np.pi) % (2 * np.pi) - np.pi
        df = df[df["epoch"] == df["epoch"].max()]

        # TODO: update this code with whats in notebook

        # TODO: non-crazy 2d histogram grid weighting of attributes by presence

        # TODO; resolve (kinda already fixed)
        #   ValueError: Cannot add the same path twice: media/images/c0451f6d.png
        #   Caused by logging same image via table and

        # TODO: run mnist augmentations experiments

        dfs_by_mode = {mode: df for mode, df in df.groupby('mode')}

        for mode, df in dfs_by_mode.items():
            fig, plt = pdf_grid_plot(
                df, title=f"{mode} capsule activation distributions",
                num_caps = self.args.pcae.num_caps, num_objs = self.n_classes
            )
            im1 = fig_to_wandb_im(fig)

            fig, plt = pdf_grid_plot_expanded(
                df, title=f"{mode} capsule activation distributions - extended plot",
                num_caps=self.args.pcae.num_caps, num_objs=self.n_classes
            )
            im2 = fig_to_wandb_im(fig)

            wandb.log({f"result_pose_dists/{mode}": im1,
                       f"result_pose_dists_expanded/{mode}": im2})


    def upload_tables(self):
        # TODO: both tables
        # assert hasattr(self, "img_table") and hasattr(self, "scalar_table"), "tables not initialized"
        t = time.time()
        self.vis_tables()
        print(f"took {time.time() - t}s to generate and log PLOTS to run")

        run = self.logger.experiment

        timestamp = datetime.datetime.now().strftime("day%m-%d_time%H-%M")
        artifact = wandb.Artifact(f"{timestamp}_run{run.id}", type="run")
        # artifact.add(self.img_table, "img_predictions")
        artifact.add(self.scalar_table, "scalar_predictions")
        artifact.add(self.result_table, "results")

        t = time.time()
        run.log_artifact(artifact)
        print(f"took {time.time() - t}s to log ARTIFACT to run")

        # t = time.time()
        # run.log({"img_predictions": self.img_table})
        # print(f"took {time.time() - t}s to log IMAGE TABLE to run")

        t = time.time()
        run.log({"scalar_predictions": self.scalar_table})
        print(f"took {time.time() - t}s to log SCALAR TABLE to run")

        t = time.time()
        run.log({"results": self.result_table})
        print(f"took {time.time() - t}s to log RESULTS TABLE to run")
