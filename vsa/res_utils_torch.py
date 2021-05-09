import torch
import numpy as np
import matplotlib.pyplot as plt

# This is a translation from numpy to torch, numpy code was written by Edward Frady: http://epaxon.github.io/
# TODO: Convert loops to torch

def clip(img):
    cimg = img.copy()
    cimg = torch.where(cimg > 1, 1, -1)
    return cimg


def norm_range(v):
    m = torch.min(v)
    return (v - m) / (torch.max(v) - m)


def fhrr_vec(D, N):
    if D == 1:
        # pick a random phase
        rphase = 2 * np.pi * torch.randn(N // 2)
        fhrrv = torch.zeros(2 * (N // 2))
        fhrrv[:(N // 2)] = torch.cos(rphase)
        fhrrv[(N // 2):] = torch.sin(rphase)
        return fhrrv

    # pick a random phase
    rphase = 2 * np.pi * torch.randn(D, N // 2)

    fhrrv = torch.zeros((D, 2 * (N // 2)))
    fhrrv[:, :(N // 2)] = torch.cos(rphase)
    fhrrv[:, (N // 2):] = torch.sin(rphase)

    return fhrrv


def cdot(v1, v2):
    return torch.dot(torch.real(v1), torch.real(v2)) + torch.dot(torch.imag(v1), torch.imag(v2))


def cvec(N, D=1):
    rphase = 2 * np.pi * torch.randn(N)
    if D == 1:
        return torch.cos(rphase) + 1.0j * torch.sin(rphase)
    vecs = torch.zeros((D, N), dtype=torch.cfloat)
    for i in range(D):
        vecs[i] = torch.cos(rphase * (i + 1)) + 1.0j * torch.sin(rphase * (i + 1))
    return vecs


def crvec(N, D=1):
    rphase = 2 * np.pi * torch.randn(D, N)
    return torch.cos(rphase) + 1.0j * torch.sin(rphase)


def cvecff(N, D, iff=1, iNf=None):
    if iNf is None:
        iNf = N

    rphase = 2 * np.pi * torch.randint(N // iff, size=(N, D)) / iNf
    return torch.cos(rphase) + 1.0j * torch.sin(rphase)


def inv_hyper(v):
    conj = torch.conj(v)
    inv = conj / torch.abs(conj)
    return inv


# D = (number x color x position)
def res_codebook_cts(N=10000, D=(180, 180, 80)):
    vecs = []

    for iD, Dv in enumerate(D):
        # v = 2 * (np.random.randn(Dv, N) < 0) - 1
        v = cvec(N, Dv)

        # stack the identity vector
        cv = cvec(N, 1)
        cv[:] = 1.5
        v = torch.vstack((v, cv))

        vecs.append(v)

    return vecs


# D = (number x color x position)
def res_codebook_bin(N=10000, D=(180, 180, 80)):
    vecs = []

    for iD, Dv in enumerate(D):
        v = 2 * (torch.randn(Dv, N) < 0) - 1

        # stack the identity vector
        cv = torch.ones(N, 1)
        v = torch.vstack((v, cv))

        vecs.append(v)

    return vecs


def make_sparse_ngram_vec(probs, vecs):
    N = vecs[0].shape[1]
    mem_vec = torch.zeros(N).astype(torch.cfloat)
    sparse_ngrams = len(probs) * [0]

    for ip, pv in enumerate(probs):
        bv = np.ones(N).astype(torch.cfloat)

        ic_idxs = len(vecs) * [0]

        for iD in range(len(vecs)):
            Dv = vecs[iD].shape[0]

            ic_idxs[iD] = torch.randint(Dv)

            i_coefs = torch.zeros(Dv).astype(torch.cfloat)
            i_coefs[ic_idxs[iD]] = 1.0

            bv *= torch.dot(i_coefs, vecs[iD])

        mem_vec += pv * bv
        sparse_ngrams[ip] = ic_idxs

    return mem_vec, sparse_ngrams


def make_sparse_continuous_ngram_vec(probs, vecs):
    N = vecs[0].shape[1]
    mem_vec = torch.zeros(N).astype(torch.cfloat)
    sparse_ngrams = len(probs) * [0]

    for ip, pv in enumerate(probs):
        bv = np.ones(N).astype(torch.cfloat)

        ic_idxs = len(vecs) * [0]

        for iD in range(len(vecs)):
            Dv = vecs[iD].shape[0]

            ic_idxs[iD] = (Dv - 2) * torch.rand() + 1

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
        x_st = x_st / torch.norm(x_st)
        x_states.append(x_st)

        x_hi = torch.zeros((max_steps, Dv))
        x_hists.append(x_hi)

    for i in range(max_steps):
        th_vec = bound_vec.copy()
        all_converged = torch.zeros(len(vecs))
        for iD in range(len(vecs)):
            x_hists[iD][i, :] = torch.real(torch.dot(torch.conj(vecs[iD]), x_states[iD]))

            if i > 1:
                all_converged[iD] = torch.allclose(x_hists[iD][i, :], x_hists[iD][i - 1, :],
                                                atol=5e-3, rtol=2e-2)

            xidx = torch.argmax(torch.abs(torch.real(x_hists[iD][i, :])))
            x_states[iD] *= torch.sign(x_hists[iD][i, xidx])

            th_vec *= torch.conj(x_states[iD])

        if torch.all(all_converged):
            print('converged:', i)
            break

        for iD in range(len(vecs)):
            x_upd = th_vec / torch.conj(x_states[iD])

            x_upd = torch.dot(vecs[iD].T, torch.real(torch.dot(torch.conj(vecs[iD]), x_upd)))

            x_states[iD] = x_upd / torch.norm(x_upd)

    return x_hists, i


def res_decode_abs(bound_vec, vecs, max_steps=100):
    x_states = []
    x_hists = []

    for iD in range(len(vecs)):
        N = vecs[iD].shape[1]
        Dv = vecs[iD].shape[0]

        x_st = cvec(N, 1)
        x_st = x_st / torch.norm(x_st)
        x_states.append(x_st)

        x_hi = torch.zeros((max_steps, Dv))
        x_hists.append(x_hi)

    for i in range(max_steps):
        th_vec = bound_vec.copy()
        all_converged = torch.zeros(len(vecs))
        for iD in range(len(vecs)):
            x_hists[iD][i, :] = torch.real(torch.dot(torch.conj(vecs[iD]), x_states[iD]))

            if i > 1:
                all_converged[iD] = torch.allclose(x_hists[iD][i, :], x_hists[iD][i - 1, :],
                                                atol=5e-3, rtol=2e-2)

            xidx = torch.argmax(torch.abs(np.real(x_hists[iD][i, :])))
            x_states[iD] *= torch.sign(x_hists[iD][i, xidx])

            th_vec *= torch.conj(x_states[iD])

        if torch.all(all_converged):
            print('converged:', i)
            break

        for iD in range(len(vecs)):
            x_upd = th_vec / torch.conj(x_states[iD])

            x_upd = torch.dot(vecs[iD].T, torch.real(torch.dot(torch.conj(vecs[iD]), x_upd)) / N)

            x_states[iD] = x_upd / torch.abs(x_upd)

    return x_hists, i


def res_decode_exaway(bound_vec, vecs, max_steps=100):
    x_states = []
    x_hists = []

    for iD in range(len(vecs)):
        N = vecs[iD].shape[1]
        Dv = vecs[iD].shape[0]

        x_st = cvec(N, 1)
        x_st = x_st / torch.norm(x_st)
        x_states.append(x_st)

        x_hi = torch.zeros((max_steps, Dv))
        x_hists.append(x_hi)

    for i in range(max_steps):
        th_vec = bound_vec.copy()
        all_converged = torch.zeros(len(vecs))
        for iD in range(len(vecs)):
            x_hists[iD][i, :] = torch.real(torch.dot(torch.conj(vecs[iD]), x_states[iD]))

            if i > 1:
                all_converged[iD] = torch.allclose(x_hists[iD][i, :], x_hists[iD][i - 1, :],
                                                atol=5e-3, rtol=2e-2)

            xidx = torch.argmax(torch.abs(torch.real(x_hists[iD][i, :])))
            x_states[iD] *= torch.sign(x_hists[iD][i, xidx])

            th_vec *= torch.conj(x_states[iD])

        if torch.all(all_converged):
            print('converged:', i)
            break

        for iD in range(len(vecs)):
            x_upd = th_vec / torch.conj(x_states[iD])

            x_upd = torch.dot(vecs[iD].T, torch.real(torch.dot(torch.conj(vecs[iD]), x_upd)))

            x_states[iD] = x_upd / torch.norm(x_upd)

    return x_hists, i


def get_output_conv(coef_hists, nsteps=None):
    alphis = []
    fstep = coef_hists[0].shape[0]

    for i in range(len(coef_hists)):
        if nsteps is None:
            alphis.append(torch.argmax(torch.abs(coef_hists[i][-1, :])))
        else:
            alphis.append(torch.argmax(torch.abs(coef_hists[i][nsteps, :])))
            fstep = nsteps

    for st in range(fstep - 1, 0, -1):
        aa = []
        for i in range(len(coef_hists)):
            aa.append(torch.argmax(torch.abs(coef_hists[i][st, :])))

        if not alphis == aa:
            break

    return alphis, st

def svd_whiten(X):
    U, s, V = torch.svd(X.reshape(X.shape[0], -1), compute_uv=True)
    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    return (U @ V.T).reshape(X.shape)

def complex_abs(vec):
    """
    Computes vector of norms for each complex number in input
    """
    return torch.sqrt(torch.conj(vec) * vec)

# KEEP IN NUMPY
def resplot_im(coef_hists, nsteps=None, vals=None, labels=None, ticks=None):
    alphis = []
    for i in range(len(coef_hists)):
        if nsteps is None:
            alphis.append(torch.argmax(torch.abs(coef_hists[i][-1, :])))
        else:
            alphis.append(torch.argmax(torch.abs(coef_hists[i][nsteps, :])))

    rows = 1
    columns = len(coef_hists)

    fig = plt.gcf();
    ax = columns * [0]

    for j in range(columns):
        ax[j] = fig.add_subplot(rows, columns, j + 1)
        if nsteps is not None:
            a = np.sign(coef_hists[j][nsteps, alphis[j]])
            coef_hists[j][:, alphis[j]] *= a

            x_h = coef_hists[j][:nsteps, :]
        else:
            a = np.sign(coef_hists[j][-1, alphis[j]])
            coef_hists[j][:, alphis[j]] *= a

            x_h = coef_hists[j][:, :]

        imh = ax[j].imshow(x_h, interpolation='none', aspect='auto')

        if j == 0:
            ax[j].set_ylabel('Iterations')
        else:
            ax[j].set_yticks([])

        if labels is not None:
            ax[j].set_title(labels[j][alphis[j]])

            if ticks is not None:
                ax[j].set_xticks(ticks[j])
                ax[j].set_xticklabels(labels[j][ticks[j]])
            else:
                ax[j].set_xticklabels(labels[j])

        elif vals is not None:
            dot_val = np.dot(x_h[-1, :], vals[j])
            ax[j].set_title(dot_val)
            # ax.set_title(vals[j][alphis[j]])

            if ticks is not None:
                ax[j].set_xticks(ticks[j])
                ax[j].set_xticklabels(vals[j][ticks])
            else:
                ax[j].set_xticklabels(vals[j])
        else:
            ax[j].set_title(alphis[j])

    # colorbar(imh, ticks=[])

    plt.tight_layout()