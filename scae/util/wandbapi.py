import wandb
from pytorch_lightning.utilities.cloud_io import load as pl_load


def download_model(run_path):
    best_model = wandb.restore('last.ckpt', run_path=run_path, replace=True)
    return pl_load(best_model.name)
