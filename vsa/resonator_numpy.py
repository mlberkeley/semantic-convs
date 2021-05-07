import numpy as np
from PIL import ImageFont
import matplotlib.pyplot as plt
from res_utils_numpy import *

patch_size=[56, 56]


def encode_pix(im, Vt, Ht):
    N = Vt.shape[0]

    image_vec = 0.0 * cvecl(N, 1)

    for m in range(im.shape[0]):
        for n in range(im.shape[1]):
            P_vec = (Vt ** m) * (Ht ** n)

            image_vec += P_vec * im[m, n]

    return image_vec

def encode_pix_rgb(im, Vt, Ht, Cv):
    N = Vt.shape[0]

    image_vec = 0.0 * cvecl(N, 1)

    for m in range(im.shape[0]):
        for n in range(im.shape[1]):
            for c in range(im.shape[2]):
                P_vec = Cv[c] * (Vt ** m) * (Ht ** n)

                image_vec += P_vec * im[m, n, c]

    return image_vec

def decode_pix(image_vec, Vt, Ht):
    N = Vt.shape[0]
    im_r = np.zeros(patch_size)

    for m in range(im_r.shape[0]):
        for n in range(im_r.shape[1]):
            P_vec = (Vt ** m) * (Ht ** n)
            im_r[m, n] = np.real(np.dot(np.conj(P_vec), image_vec) / N)
    return im_r


def decode_pix_rgb(image_vec, Vt, Ht, Cv):
    N = Vt.shape[0]
    im_r = np.zeros(fim_size)

    for m in range(im_r.shape[0]):
        for n in range(im_r.shape[1]):
            for c in range(im_r.shape[2]):
                P_vec = Cv[c] * (Vt ** m) * (Ht ** n)
                im_r[m, n, c] = np.real(np.dot(np.conj(P_vec), image_vec) / N)
    return np.clip(im_r, 0, 1)


if __name__ == "__main__":
    font = ImageFont.truetype(u'/usr/share/fonts/opentype/mathjax/MathJax_Typewriter-Regular.otf', size=18)
    letters = 'abcde'#fghijklmnopqrstuvwxyz'

    fim_size = (patch_size[0], patch_size[1], 1)
    font_ims = []
    for l in letters:
        font_obj = font.getmask(l)

        imtext = np.array(font_obj)
        imsize = font_obj.size  # font.getsize(l)

        imtext = np.tile(imtext.reshape((imsize[1], imsize[0], 1)), (1, 1, 1))
        imtext = imtext[:patch_size[0], :patch_size[1]]

        imsize = imtext.shape

        fim = np.zeros(fim_size)

        fimr = int(np.floor((fim_size[0] - imsize[0]) / 2))
        fimc = int(np.floor((fim_size[1] - imsize[1]) / 2))

        fim[fimr:(fimr + imsize[0]), fimc:(fimc + imsize[1])] = imtext / 255

        font_ims.append(fim)
    # plt.imshow(fim)
    # plt.show()


    N = int(3e4)

    # These are special base vectors for position that loop
    Vt = cvecl(N, font_ims[0].shape[0])
    Ht = cvecl(N, font_ims[0].shape[1])

    # This is a set of 3 independently random complex phasor vectors for color
    # Cv = crvec(N, 3)

    font_im_vecs = np.zeros((len(font_ims), np.prod(font_ims[0].shape[:2])))
    for i in range(len(font_ims)):
        font_im_vecs[i, :] = font_ims[i].mean(axis=2).ravel()
    font_ims_w = svd_whiten(font_im_vecs.T).T

    # font_vecs = crvec(N, len(font_ims))
    # for i in range(len(font_ims)):
    #     print(i, end=" ")
    #     font_vecs[i, :] = encode_pix(font_ims[i].mean(axis=2), Vt, Ht)
    font_vecs_w = crvec(N, len(font_ims))
    for i in range(len(font_ims)):
        print(i, end=" ")
        font_vecs_w[i, :] = encode_pix(font_ims_w[i].reshape(font_ims[0].shape[:2]), Vt, Ht)

    Vspan = font_ims[0].shape[0]
    Hspan = font_ims[0].shape[1]
    Vt_span = crvec(N, Vspan)
    Ht_span = crvec(N, Hspan)
    for i in range(Vspan):
        ttV = i - Vspan // 2
        Vt_span[i, :] = Vt ** ttV
    for i in range(Hspan):
        ttH = i - Hspan // 2
        Ht_span[i, :] = Ht ** ttH

    res_vecs = []
    # res_vecs.append(color_vecs_w)
    res_vecs.append(font_vecs_w)
    res_vecs.append(Vt_span)
    res_vecs.append(Ht_span)

    # Generate a scene of three objects with random factors

    im_idx1 = np.random.randint(len(font_ims))
    tH = patch_size[1] * np.random.rand(1)
    tV = patch_size[0] * np.random.rand(1)
    # rC = np.random.randint(colors_arr.shape[0])

    t_im1 = font_ims[im_idx1].copy()

    # for i in range(t_im1.shape[2]):
    #     t_im1[:, :, i] = colors_arr[rC, i] * t_im1[:, :, i]
    from scipy.ndimage.interpolation import shift
    t_im1 = shift(t_im1, (tV, tH, 0), mode='wrap', order=1)

    t_im = np.clip(t_im1, 0, 1)

    plt.imshow(t_im, interpolation='none')
    plt.show()

    bound_vec = encode_pix(t_im, Vt, Ht)
    res_hist, nsteps = res_decode_abs(bound_vec, res_vecs, 200)

    resplot_im(res_hist, nsteps)
    plt.show()
    pass