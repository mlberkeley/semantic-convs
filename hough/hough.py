from scae.data import letters
from scae.util.plots import plot_image_tensor, plot_image_tensor_2D, fig_to_wandb_im
import torchshow

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

import wandb
from pytorch_lightning.loggers import WandbLogger


class HoughFeatures(nn.Module):
    def __init__(self, latent_dim, img_shape=(28 ,28)):
        super().__init__()
        self.img_shape = img_shape
        self.backbone = nn.Sequential(
            nn.Conv2d(1, latent_dim, kernel_size=(7 ,7), stride=(1 ,1), padding=(3 ,3), bias=False),
            nn.BatchNorm2d(latent_dim),
            # nn.ReLU(),
        )

    def forward(self, x):
        z = self.backbone(x)
        return z

class HoughFeatureMatcher(pl.LightningModule):

    def __init__(self, template_shape=(28 ,28), num_templates=10, num_feats=4):
        super().__init__()
        self.num_templates = num_templates

        self.feats = HoughFeatures(num_feats)
        self.templates = nn.Parameter(torch.rand(num_templates, 1, *template_shape))

        self.table = []

    def forward(self, x, vis=False):
        """ I = T  """
        x_feats, t_feats = self.feats(x), self.feats(self.templates)

        # Dot product over last 3 dims (C,H,W)
        #         matches = torch.einsum("bchw,tchw->bt", x_feats, t_feats)
        #         matches = F.softmax(matches, dim=-1)
        #         rec = torch.einsum("bt,tchw->bchw", matches, self.templates)

        #         matches = torch.einsum("bchw,tchw->bt", x_feats, t_feats)

        matches = F.conv2d(x_feats, t_feats)

        matches_shape = matches.shape
        # Softmax over template parameter-activation maps (one template active)
        # matches = F.softmax(matches.view(matches_shape[0], -1), dim=-1).view(matches_shape)
        # Softmax over template parameter-activation maps (individually)
        matches = F.softmax(matches.view(*matches_shape[:2], -1), dim=-1).view(matches_shape)

        rec = F.conv_transpose2d(matches, self.templates)#, groups=self.num_templates)
        #         rec = torch.einsum("bt,tchw->bchw", matches, self.templates)

        if getattr(self, "table") is not None and self.current_epoch % 100 == 0:
            print(x_feats.shape, t_feats.shape, matches.shape, self.templates.shape)  # , rec.shape)
            row = {}

            fig, plt = plot_image_tensor_2D(x[None, ...], title="input_images")
            row["input_images"] = fig_to_wandb_im(fig)
            fig, plt = plot_image_tensor_2D(self.templates[None, ...], title="template_images")
            row["template_images"] = fig_to_wandb_im(fig)

            fig, plt = plot_image_tensor_2D(torch.swapaxes(x_feats, 1, 0)[:, :, None, ...], title="input_features")
            row["input_features"] = fig_to_wandb_im(fig)
            fig, plt = plot_image_tensor_2D(torch.swapaxes(t_feats, 1, 0)[:, :, None, ...], title="template_features")
            row["template_features"] = fig_to_wandb_im(fig)

            fig, plt = plot_image_tensor_2D(matches[:, :, None, ...], title="matches", titles=[f"tmp{i}" for i in range(self.num_templates)])
            row["matches"] = fig_to_wandb_im(fig)

            # torchshow.show(matches.view(*matches.shape[:-2], -1).sum(-1))

            fig, plt = plot_image_tensor_2D(rec[None, ...], title="reconstruction")
            row["reconstruction"] = fig_to_wandb_im(fig)

            self.table.append(row)

        return rec

    def training_step(self, batch, batch_idx):
        x = batch
        rec = self(x)
        loss = F.mse_loss(x, rec)
        trainer.logger.experiment.log({
            "loss": loss,
            "epoch": self.current_epoch
        })
        self.table[-1]["batch_idx"] = batch_idx
        self.table[-1]["loss"] = loss
        self.table[-1]["epoch"] = self.current_epoch
        return loss

    def on_train_end(self):
        wandb_table = wandb.Table(columns=list(self.table[0].keys()))
        for row in self.table:
            wandb_table.add_data(*list(row.values()))
        self.logger.experiment.log({"training_table": wandb_table})
        self.table = []

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(recurse=True), lr=0.1)
        lr_sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=.996)
        return [opt], [lr_sched]


if __name__ == "__main__":
    # num_letters = 6
    # dset_notransform = letters.Letters(num_letters=num_letters)
    # gt_temp_shape = dset_notransform[0].shape
    # print("gt_temp_shape", gt_temp_shape)
    #
    # transform = transforms.Compose([
    #     transforms.RandomCrop(size=(40,40), pad_if_needed=True),
    #     # norm_1c
    # ])
    # dset = letters.Letters(num_letters=num_letters, transform=transform)
    # print(dset.images.shape)
    # plt, _ = plot_image_tensor(dset.images)
    # plt.show()

    # Sanity check imp
    # model = HoughFeatureMatcher(template_shape=dset.images.shape[-2:], num_templates=num_letters)

    # model(dset[:model.num_templates], vis=True)
    # import time; time.sleep(100)
    # model.templates = torch.nn.Parameter(dset[:model.num_templates])
    # model(dset[:model.num_templates], vis=True)

    # Train
    num_letters = 6
    template_shape = (15, 15)  # dset.images.shape[-2:],
    num_templates = 6
    num_feats = 4

    transform = transforms.Compose([
        transforms.RandomCrop(size=(40, 40), pad_if_needed=True),
    ])
    dset = letters.Letters(num_letters=num_letters, transform=transform)
    dloader = DataLoader(dset, batch_size=num_letters*10)

    model = HoughFeatureMatcher(
        template_shape=template_shape,  # dset.images.shape[-2:],
        num_templates=num_templates,  # num_letters,
        num_feats=num_feats,
    )

    logger = WandbLogger(
        project="StackedCapsuleAutoEncoders",
        name=f"n_letters:{num_letters} tmp_shp:{template_shape} n_tmp:{num_templates} n_feats:{num_feats}",
        entity="mlatberkeley",
    )
    trainer = pl.Trainer(
        max_epochs=2000,
        progress_bar_refresh_rate=50,
        gpus=1,
        logger=logger,
    )
    trainer.fit(model, dloader)

    # Vis result of training
    out = model(dset[:model.num_templates], vis=True)
    pass