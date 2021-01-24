import torch
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


def plot_template_assembly_grid(X):
    plt.figure()
    num_plot = 5
    fig, ax = plt.subplots(num_plot, num_plot)
    for i in range(num_plot):
        for j in range(num_plot):
            idx = np.random.randint(0, X.shape[0])
            ax[i, j].imshow(X[idx])
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
    fig.subplots_adjust(hspace=0.1)  # No horizontal space between subplots
    fig.subplots_adjust(wspace=0)
    fig.show()


def print_edict(edict: EasyDict, prefix=''):
    if not prefix:
        print(f'# EasyDict object id {id(edict)}')
    for k in edict:
        v = edict[k]
        if isinstance(v, EasyDict):
            print(f'\n{prefix}{k}:')
            print_edict(v, prefix=f'  {prefix}')
        else:
            print(f'{prefix}{k}: {v}')
    if not prefix:
        print()

# with open('scae/save.pkl', 'rb') as input:
#     capsules_l = pickle.load(input)
#     rec_l = pickle.load(input)
