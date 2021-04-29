from __future__ import division

from pylab import *
import scipy
import time

import sklearn
from sklearn.decomposition import PCA, FastICA, TruncatedSVD, NMF

import colormaps

plt.rcParams.update({'axes.titlesize': 'xx-large'})
plt.rcParams.update({'axes.labelsize': 'xx-large'})
plt.rcParams.update({'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large'})
plt.rcParams.update({'legend.fontsize': 'x-large'})
plt.rcParams.update({'text.usetex': True})


def clip(img):
    cimg = img.copy()
    cimg[cimg > 1] = 1
    cimg[cimg < 1] = -1
    return cimg


def norm_range(v):
    return (v - v.min()) / (v.max() - v.min())


def svd_whiten(X):
    U, s, Vh = np.linalg.svd(X, full_matrices=False)

    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    X_white = np.dot(U, Vh)

    return X_white


def fhrr_vec(D, N):
    if D == 1:
        # pick a random phase
        rphase = 2 * np.pi * np.random.rand(N // 2)
        fhrrv = np.zeros(2 * (N // 2))
        fhrrv[:(N // 2)] = np.cos(rphase)
        fhrrv[(N // 2):] = np.sin(rphase)
        return fhrrv

    # pick a random phase
    rphase = 2 * np.pi * np.random.rand(D, N // 2)

    fhrrv = np.zeros((D, 2 * (N // 2)))
    fhrrv[:, :(N // 2)] = np.cos(rphase)
    fhrrv[:, (N // 2):] = np.sin(rphase)

    return fhrrv


def cdot(v1, v2):
    return np.dot(np.real(v1), np.real(v2)) + np.dot(np.imag(v1), np.imag(v2))


def cvec(N, D=1):
    rphase = 2 * np.pi * np.random.rand(N)
    if D == 1:
        return np.cos(rphase) + 1.0j * np.sin(rphase)
    vecs = np.zeros((D, N), 'complex')
    for i in range(D):
        vecs[i] = np.cos(rphase * (i + 1)) + 1.0j * np.sin(rphase * (i + 1))
    return vecs


def crvec(N, D=1):
    rphase = 2 * np.pi * np.random.rand(D, N)
    return np.cos(rphase) + 1.0j * np.sin(rphase)


def roots(z, n):
    nthRootOfr = np.abs(z) ** (1.0 / n)
    t = np.angle(z)
    return map(lambda k: nthRootOfr * np.exp((t + 2 * k * pi) * 1j / n), range(n))


def cvecl(N, loopsize=None):
    if loopsize is None:
        loopsize = N

    unity_roots = np.array(list(roots(1.0 + 0.0j, loopsize)))
    root_idxs = np.random.randint(loopsize, size=N)
    X1 = unity_roots[root_idxs]

    return X1


def cvecff(N, D, iff=1, iNf=None):
    if iNf is None:
        iNf = N

    rphase = 2 * np.pi * np.random.randint(N // iff, size=(N, D)) / iNf
    return np.cos(rphase) + 1.0j * np.sin(rphase)


def inv_hyper(v):
    conj = np.conj(v)
    inv = conj / np.abs(conj)
    return inv


# D = (number x color x position)
def res_codebook_cts(N=10000, D=(180, 180, 80)):
    vecs = []

    for iD, Dv in enumerate(D):
        # v = 2 * (np.random.randn(Dv, N) < 0) - 1
        v = cvec(N, Dv).T

        # stack the identity vector
        cv = cvec(N, 1)
        cv[:] = 1.5
        v = np.vstack((v, cv))

        vecs.append(v)

    return vecs


# D = (number x color x position)
def res_codebook_bin(N=10000, D=(180, 180, 80)):
    vecs = []

    for iD, Dv in enumerate(D):
        v = 2 * (np.random.randn(Dv, N) < 0) - 1

        # stack the identity vector
        cv = np.ones(N, 1)
        v = np.vstack((v, cv))

        vecs.append(v)

    return vecs


def make_sparse_ngram_vec(probs, vecs):
    N = vecs[0].shape[1]
    mem_vec = np.zeros(N).astype('complex')
    sparse_ngrams = len(probs) * [0]

    for ip, pv in enumerate(probs):
        bv = np.ones(N).astype('complex')

        ic_idxs = len(vecs) * [0]

        for iD in range(len(vecs)):
            Dv = vecs[iD].shape[0]

            ic_idxs[iD] = np.random.randint(Dv)

            i_coefs = np.zeros(Dv).astype('complex')
            i_coefs[ic_idxs[iD]] = 1.0

            bv *= np.dot(i_coefs, vecs[iD])

        mem_vec += pv * bv
        sparse_ngrams[ip] = ic_idxs

    return mem_vec, sparse_ngrams


def make_sparse_continuous_ngram_vec(probs, vecs):
    N = vecs[0].shape[1]
    mem_vec = np.zeros(N).astype('complex')
    sparse_ngrams = len(probs) * [0]

    for ip, pv in enumerate(probs):
        bv = np.ones(N).astype('complex')

        ic_idxs = len(vecs) * [0]

        for iD in range(len(vecs)):
            Dv = vecs[iD].shape[0]

            ic_idxs[iD] = (Dv - 2) * np.random.rand() + 1

            bv *= vecs[iD][0, :] ** ic_idxs[iD]
            # bv *= np.dot(i_coefs, vecs[iD])

        mem_vec += pv * bv
        sparse_ngrams[ip] = ic_idxs

    return mem_vec, sparse_ngrams


def res_decode(bound_vec, vecs, max_steps=100):
    x_states = []
    x_hists = []

    for iD in range(len(vecs)):
        N = vecs[iD].shape[1]
        Dv = vecs[iD].shape[0]

        x_st = cvec(N, 1)
        x_st = x_st / np.linalg.norm(x_st)
        x_states.append(x_st)

        x_hi = np.zeros((max_steps, Dv))
        x_hists.append(x_hi)

    for i in range(max_steps):
        th_vec = bound_vec.copy()
        all_converged = np.zeros(len(vecs))
        for iD in range(len(vecs)):
            x_hists[iD][i, :] = np.real(np.dot(np.conj(vecs[iD]), x_states[iD]))

            if i > 1:
                all_converged[iD] = np.allclose(x_hists[iD][i, :], x_hists[iD][i - 1, :],
                                                atol=5e-3, rtol=2e-2)

            xidx = np.argmax(np.abs(np.real(x_hists[iD][i, :])))
            x_states[iD] *= np.sign(x_hists[iD][i, xidx])

            th_vec *= np.conj(x_states[iD])

        if np.all(all_converged):
            print('converged:', i, end=" ")
            break

        for iD in range(len(vecs)):
            x_upd = th_vec / np.conj(x_states[iD])

            x_upd = np.dot(vecs[iD].T, np.real(np.dot(np.conj(vecs[iD]), x_upd)))

            x_states[iD] = x_upd / np.linalg.norm(x_upd)

    return x_hists, i


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


def res_decode_abs_exaway(bound_vec, vecs, max_steps=100, x_hi_init=None):
    x_states = []
    x_hists = []
    ra_hist = []
    vecsw = []

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

        vecsw.append(svd_whiten(vecs[iD]))

    for i in range(max_steps):

        res_recon = crvec(N, 1) ** 0

        for iD in range(len(vecs)):
            rr = np.dot(vecs[iD].T, np.real(np.dot(np.conj(vecs[iD]), x_states[iD])))
            rr /= np.abs(rr)

            res_recon *= rr

        # res_recon = np.prod(x_states)
        res_alpha = cdot(res_recon, bound_vec) / N
        ra_hist.append(res_alpha)

        th_vec = bound_vec.copy() - res_alpha * res_recon

        all_converged = np.zeros(len(vecs))

        th_vec *= np.conj(res_recon)

        # rr2 = np.prod(x_states)
        # th_vec *= np.conj(rr2)

        # for iD in range(len(vecs)):
        # if i > 1:
        #    xidx = np.argmax(np.abs(np.real(x_hists[iD][i-1, :])))
        #    x_states[iD] *= np.sign(x_hists[iD][i-1, xidx])

        # th_vec *= np.conj(x_states[iD])

        for iD in range(len(vecs)):
            x_upd = th_vec / np.conj(x_states[iD])

            x_upd = np.dot(vecsw[iD].T, np.real(np.dot(np.conj(vecsw[iD]), x_upd.T)))
            # x_upd = np.dot(vecs[iD].T, np.dot(np.conj(vecs[iD]), x_upd))

            # x_states[iD] = 0.85*(x_upd / np.abs(x_upd)) + 0.15*x_states[iD]
            # x_states[iD] +=
            x_states[iD] += (x_upd / np.abs(x_upd))
            x_states[iD] /= np.abs(x_states[iD])

            x_hists[iD][i, :] = np.real(np.dot(np.conj(vecs[iD]), x_states[iD]))

            if i > 1:
                all_converged[iD] = np.allclose(x_hists[iD][i, :], x_hists[iD][i - 1, :],
                                                atol=5e-3, rtol=2e-2)

        if np.all(all_converged):
            print('converged:', i, end=" ")
            break

    return x_hists, i, ra_hist


def res_decode_exaway(bound_vec, vecs, max_steps=100, x_hi_init=None):
    x_states = []
    x_hists = []

    for iD in range(len(vecs)):
        N = vecs[iD].shape[1]
        Dv = vecs[iD].shape[0]

        if x_hi_init is None:
            x_st = crvec(N, 1)
            x_st = x_st / np.abs(x_st)
        else:
            x_st = np.dot(vecs[iD], x_hi_init[iD])

        x_states.append(x_st)

        x_hi = np.zeros((max_steps, Dv))
        x_hists.append(x_hi)

    for i in range(max_steps):
        th_vec = bound_vec.copy()

        all_converged = np.zeros(len(vecs))
        for iD in range(len(vecs)):
            x_hists[iD][[i], :] = np.real(np.dot(np.conj(vecs[iD]), x_states[iD].T) / N).T

            if i > 1:
                all_converged[iD] = np.allclose(x_hists[iD][i, :], x_hists[iD][i - 1, :],
                                                atol=5e-3, rtol=2e-2)

            # xidx = np.argmax(np.abs(np.real(x_hists[iD][i, :])))
            # x_states[iD] *= np.sign(x_hists[iD][i, xidx])

            th_vec *= np.conj(x_states[iD])

        if np.all(all_converged):
            print('converged:', i, end=" ")
            break

        for iD in range(len(vecs)):
            x_upd = th_vec / np.conj(x_states[iD])

            x_upd = np.dot(vecs[iD].T, np.real(np.dot(np.conj(vecs[iD]), x_upd.T))).T / N

            x_states[iD] += 0.9 * x_upd

    return x_hists, i


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

    fig = gcf();
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

        imh = ax[j].imshow(x_h, interpolation='none', aspect='auto', cmap=colormaps.viridis)

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


def get_output_conv(coef_hists, nsteps=None):
    alphis = []
    fstep = coef_hists[0].shape[0]

    for i in range(len(coef_hists)):
        if nsteps is None:
            alphis.append(np.argmax(np.abs(coef_hists[i][-1, :])))
        else:
            alphis.append(np.argmax(np.abs(coef_hists[i][nsteps, :])))
            fstep = nsteps

    for st in range(fstep - 1, 0, -1):
        aa = []
        for i in range(len(coef_hists)):
            aa.append(np.argmax(np.abs(coef_hists[i][st, :])))

        if not alphis == aa:
            break

    return alphis, st
