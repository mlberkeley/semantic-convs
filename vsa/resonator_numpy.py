import numpy as np
from PIL import ImageFont
import matplotlib.pyplot as plt

patch_size=[56, 56]

def crvec(N, D=1):
    rphase = 2 * np.pi * np.random.rand(D, N)
    return np.cos(rphase) + 1.0j * np.sin(rphase)

def roots(z, n):
    nthRootOfr = np.abs(z) ** (1.0 / n)
    t = np.angle(z)
    return map(lambda k: nthRootOfr * np.exp((t + 2 * k * np.pi) * 1j / n), range(n))

def cvecl(N, loopsize=None):
    if loopsize is None:
        loopsize = N

    unity_roots = np.array(list(roots(1.0 + 0.0j, loopsize)))
    root_idxs = np.random.randint(loopsize, size=N)
    X1 = unity_roots[root_idxs]

    return X1

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

def svd_whiten(X):
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    X_white = np.dot(U, Vh)
    return X_white

def resplot_im(coef_hists, nsteps=None, vals=None, labels=None, ticks=None, gt_labels=None):
    alphis = []
    for i in range(len(coef_hists)):
        if nsteps is None:
            alphis.append(np.argmax(np.abs(coef_hists[i][-1, :])))
        else:
            alphis.append(np.argmax(np.abs(coef_hists[i][nsteps, :])))
    print(alphis)

    rows = 1
    columns = len(coef_hists)

    fig = plt.gcf()
    ax = columns * [0]

    for j in range(columns):
        ax[j] = fig.add_subplot(rows, columns, j + 1)
        if nsteps is not None:
            a = np.sign(coef_hists[j][nsteps, alphis[j]])
            coef_hists[j] *= a

            x_h = coef_hists[j][:nsteps, :]
        else:
            a = np.sign(coef_hists[j][-1, alphis[j]])
            coef_hists[j] *= a

            x_h = coef_hists[j][:, :]

        imh = ax[j].imshow(x_h, interpolation='none', aspect='auto')#, cmap=colormaps.viridis)

        if j == 0:
            ax[j].set_ylabel('Iterations')
        else:
            ax[j].set_yticks([])

        if labels is not None:
            ax[j].set_title(labels[j][alphis[j]])
            # ax[j].set_xlabel(labels[j][alphis[j]])

            if ticks is not None:
                ax[j].set_xticks(ticks[j])
                ax[j].set_xticklabels(labels[j][ticks[j]])
            else:
                ax[j].set_xticks(np.arange(len(labels[j])))
                ax[j].set_xticklabels(labels[j])

        elif vals is not None:
            dot_val = np.dot(x_h[-1, :], vals[j])
            # ax[j].set_title(dot_val)
            ax[j].set_xlabel(dot_val)

            # ax.set_title(vals[j][alphis[j]])

            if ticks is not None:
                ax[j].set_xticks(ticks[j])
                ax[j].set_xticklabels(vals[j][ticks])
            else:
                ax[j].set_xticklabels(vals[j])
        else:
            ax[j].set_title(alphis[j])
            # ax[j].set_xlabel(alphis[j])

        if gt_labels is not None:
            # ax[j].set_xlabel(gt_labels[j])
            ax[j].set_title(gt_labels[j])

    # colorbar(imh, ticks=[])

    plt.tight_layout()

def res_decode_abs(bound_vec, vecs, max_steps=100, x_hi_init=None):
    x_states = []
    x_hists = []

    for iD in range(len(vecs)):
        N = vecs[iD].shape[1]
        Dv = vecs[iD].shape[0]

        if x_hi_init is None:
            x_st = crvec(N, 1)
            x_st = np.squeeze(x_st / np.abs(x_st))
        else:
            x_st = np.dot(vecs[iD].T, x_hi_init[iD])

        x_states.append(x_st)

        x_hi = np.zeros((max_steps, Dv))
        x_hists.append(x_hi)

    for i in range(max_steps):
        th_vec = bound_vec.copy()
        all_converged = np.zeros(len(vecs))
        for iD in range(len(vecs)):
            if i > 1:
                xidx = np.argmax(np.abs(np.real(x_hists[iD][i - 1, :])))
                x_states[iD] *= np.sign(x_hists[iD][i - 1, xidx])

            th_vec *= np.conj(x_states[iD])

        for iD in range(len(vecs)):
            x_upd = th_vec / np.conj(x_states[iD])

            x_upd = np.dot(vecs[iD].T, np.real(np.dot(np.conj(vecs[iD]), x_upd)))
            # x_upd = np.dot(vecs[iD].T, np.dot(np.conj(vecs[iD]), x_upd))

            # x_states[iD] = 0.9*(x_upd / np.abs(x_upd)) + 0.1*x_states[iD]
            x_states[iD] = (x_upd / np.abs(x_upd))

            x_hists[iD][i, :] = np.real(np.dot(np.conj(vecs[iD]), x_states[iD]))

            if i > 1:
                all_converged[iD] = np.allclose(x_hists[iD][i, :], x_hists[iD][i - 1, :],
                                                atol=5e-3, rtol=2e-2)

        if np.all(all_converged):
            print('converged:', i, )
            break

    return x_hists, i

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