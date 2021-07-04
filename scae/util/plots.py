import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import wandb
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

SHOW_PLOTS = False


def fig_to_nparray(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()  # draw the canvas, cache the renderer
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return image


def fig_to_wandb_im(fig):
    image = fig_to_nparray(fig)
    return wandb.Image(image)


# TODO: these shouldn't be hardcoded
# scale_y and shear unused due to similarity constraint
# attrs = ["presence", "trans_x", "trans_y", "theta", "scale_x", "scale_y", "shear"]
attrs = ["presence", "trans_x", "trans_y", "theta", "scale_x"]


def pdf_grid_plot(df, title, num_objs, num_caps, width=20):
    len_y, len_x = num_objs + 1, len(attrs)
    fig, ax = plt.subplots(len_y, len_x, figsize=(width, width / len_x * len_y), sharex='col', facecolor='w')
    for a, attr in enumerate(attrs):
        for c in range(num_caps):
            col_str = f"c{c}_{attr}"

            caps_attr = df[col_str]
            if attr is not "presence":
                caps_presences = df[f"c{c}_presence"]
                sns.kdeplot(df[col_str], weights=caps_presences, ax=ax[0, a])
            else:
                sns.kdeplot(df[col_str], ax=ax[0, a])
            ax[0, a].set_title(f"allobjs_{attr}")

            for l in range(num_objs):
                caps_attr_given_l = caps_attr[df["label"] == l]
                if attr is not "presence":
                    caps_presences_given_l = caps_presences[df["label"] == l]
                    sns.kdeplot(caps_attr_given_l, weights=caps_presences_given_l, ax=ax[l + 1, a])
                else:
                    sns.kdeplot(caps_attr_given_l, ax=ax[l + 1, a])
                ax[l + 1, a].set_title(f"obj{l}_{attr}")

        for l in range(len_y):
            ax[l, a].set_xlabel("")
            ax[l, a].set_ylabel("")
            ax[l, a].xaxis.set_tick_params(which='both', labelbottom=True)
            ax[l, a].tick_params(
                axis='y',  # changes apply to the y-axis
                which='both',  # both major and minor ticks are affected
                left=False,  # ticks along the bottom edge are off
                right=False,  # ticks along the top edge are off
                labelleft=False)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle(title, fontsize=16)
    if SHOW_PLOTS:
        plt.show()
    return fig, plt


def pdf_grid_plot_expanded(df, title, num_objs, num_caps, width=20):
    len_y, len_x = num_objs + 1, len(attrs)
    fig = plt.figure(figsize=(width, width / len_x * len_y / 2), facecolor='w')
    ggrid = gridspec.GridSpec(len_y, len_x, figure=fig, hspace=.5)
    colormap = matplotlib.cm.get_cmap('tab10')

    for a, attr in enumerate(attrs):
        len_yy, len_xx = 2, 4
        grid_col = [
            gridspec.GridSpecFromSubplotSpec(len_yy, len_xx, subplot_spec=ggrid[l, a], wspace=0, hspace=0) for l
            in range(len_y)]
        first_axs = fig.add_subplot(grid_col[0][:, :2])
        axs_rows = [[first_axs]] + [[fig.add_subplot(grid[:, :2], sharex=first_axs)] for grid in grid_col[1:]]

        for row_idx in range(len_y):
            for yy in range(len_yy):
                for xx in range(2, len_xx):
                    ax = fig.add_subplot(grid_col[row_idx][yy, xx], sharex=first_axs)
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                    axs_rows[row_idx].append(ax)

        for c in range(num_caps):
            #         color = colormap(c / (num_caps - 1))
            color = colormap(c / 10)
            col_str = f"c{c}_{attr}"

            caps_attr = df[col_str]
            if attr is not "presence":
                caps_presences = df[f"c{c}_presence"]
                sns.kdeplot(caps_attr, weights=caps_presences, ax=axs_rows[0][0], color=color)
                sns.kdeplot(caps_attr, weights=caps_presences, ax=axs_rows[0][c + 1], color=color)
            else:
                sns.kdeplot(caps_attr, ax=axs_rows[0][0], color=color)
                sns.kdeplot(caps_attr, ax=axs_rows[0][c + 1], color=color)
            axs_rows[0][0].set_title(f"allobjs_{attr}")

            for l in range(len_y - 1):
                caps_attr_given_l = caps_attr[df["label"] == l]

                if attr is not "presence":
                    caps_presences_given_l = caps_presences[df["label"] == l]
                    sns.kdeplot(caps_attr_given_l, weights=caps_presences_given_l, ax=axs_rows[l + 1][0],
                                color=color)
                    sns.kdeplot(caps_attr_given_l, weights=caps_presences_given_l, ax=axs_rows[l + 1][c + 1],
                                color=color)
                else:
                    sns.kdeplot(caps_attr_given_l, ax=axs_rows[l + 1][0], color=color)
                    sns.kdeplot(caps_attr_given_l, ax=axs_rows[l + 1][c + 1], color=color)
                axs_rows[l + 1][0].set_title(f"obj{l}_{attr}")

    for i, ax in enumerate(fig.axes):
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(
            axis='y',  # changes apply to the y-axis
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelleft=False)
    ggrid.tight_layout(fig, rect=[0, 0, 1, 0.97])
    fig.suptitle(title, fontsize=16)
    if SHOW_PLOTS:
        plt.show()
    return fig, plt


def show_img(tensor, min=0, max=1):
    img = tensor.cpu().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(img, vmin=min, vmax=max, norm=False)
    fig.colorbar(im, ax=ax)

    if SHOW_PLOTS:
        plt.show()
    return fig, plt


def plot_image_tensor(X, width=8., title=None, titles=None):
    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X)
    num_imgs = X.shape[0]
    len_y = int(np.floor(np.sqrt(num_imgs)))
    len_x = int(np.ceil(num_imgs / len_y))

    fig, ax = plt.subplots(len_y, len_x, figsize=(width, width / len_x * len_y))
    for i in range(len_y):
        for j in range(len_x):
            a = ax[i, j] if len_y > 1 else ax[j]
            idx = i * len_x + j
            if idx < num_imgs:
                a.imshow(X[idx][0].detach().cpu().numpy())
            a.get_xaxis().set_visible(False)
            a.get_yaxis().set_visible(False)
            if titles is not None and i == 0:
                a.set_title(titles[j])
    fig.subplots_adjust(hspace=0)  # No horizontal space between subplots
    fig.subplots_adjust(wspace=0)
    fig.suptitle(title, fontsize=16)
    if SHOW_PLOTS:
        plt.show()
    return fig, plt

def plot_image_tensor_2D(X, width=8., title=None, titles=None):
    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X)
    len_y, len_x = X.shape[:2]

    fig, ax = plt.subplots(len_y, len_x, figsize=(width, width / len_x * len_y))
    for i in range(len_y):
        for j in range(len_x):
            a = ax[i, j] if len_y > 1 else ax[j]
            a.imshow(X[i, j][0].detach().cpu().numpy())
            a.get_xaxis().set_visible(False)
            a.get_yaxis().set_visible(False)
            if titles is not None and i == 0:
                a.set_title(titles[j])
    fig.subplots_adjust(hspace=0)  # No horizontal space between subplots
    fig.subplots_adjust(wspace=0)
    fig.suptitle(title, fontsize=16)
    if SHOW_PLOTS:
        plt.show()
    return fig, plt


def plot_dataloader_sorted(dset, name, num_classes=10):
    imgs, labels = next(iter(dset))
    image_grid = torch.stack([imgs[labels == c][:num_classes] for c in range(num_classes)])
    return plot_image_tensor_2D(image_grid, title=name)


def plot_dataloader(dset, name):
    imgs = next(iter(dset))[0]
    largest_square = int(imgs.shape[0] ** .5) ** 2
    return plot_image_tensor(imgs[:largest_square], title=name)


# with open('scae/save.pkl', 'rb') as input:
#     capsules_l = pickle.load(input)
#     rec_l = pickle.load(input)

