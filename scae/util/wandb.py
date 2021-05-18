import wandb
import torch


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
