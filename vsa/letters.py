import torch
import numpy as np


def gen_letter_images(patch_size = (56, 56)):
    from PIL import ImageFont
    from scae.util import vis

    font = ImageFont.truetype(u'/usr/share/fonts/opentype/mathjax/MathJax_Typewriter-Regular.otf', size=18)
    letters = 'abcdefghijklmnopqrstuvwxyz'
    font_ims = []

    fim_size = (patch_size[0], patch_size[1], 3)

    for l in letters:
        font_obj = font.getmask(l)

        imtext = np.array(font_obj)
        imsize = font_obj.size  # font.getsize(l)

        imtext = np.tile(imtext.reshape((imsize[1], imsize[0], 1)), (1, 1, 3))
        imtext = imtext[:patch_size[0], :patch_size[1], :]

        imsize = imtext.shape

        fim = np.zeros(fim_size)

        fimr = int(np.floor((fim_size[0] - imsize[0]) / 2))
        fimc = int(np.floor((fim_size[1] - imsize[1]) / 2))

        fim[fimr:(fimr + imsize[0]), fimc:(fimc + imsize[1]), :] = imtext / 255

        font_ims.append(fim)

    t_font_ims = torch.Tensor(font_ims)#.permute(0, 3, 1, 2)
    return t_font_ims