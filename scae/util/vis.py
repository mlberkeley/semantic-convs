import torch
import torchshow as ts

import matplotlib.pyplot as plt
from functools import partial
import numpy as np
from easydict import EasyDict


def show_img(tensor, min=0, max=1):
    img = tensor.cpu().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(img, vmin=min, vmax=max, norm=False)
    fig.colorbar(im, ax=ax)

    plt.show()


def plot_image_tensor(X, width=8., titles=None):
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
    fig.show()

def plot_image_tensor_2D(X, width=8., titles=None):
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
    fig.show()

# with open('scae/save.pkl', 'rb') as input:
#     capsules_l = pickle.load(input)
#     rec_l = pickle.load(input)
