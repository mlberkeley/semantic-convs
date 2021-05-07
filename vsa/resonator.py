import matplotlib.pyplot as plt
import numpy as np

import torch

from vsa import VSA, ctvec

import res_utils_torch as ru

class Resonator:
    def __init__(self, attr_dicts, device="cuda"):
        self.attr_dicts = [d.to(device) for d in attr_dicts]
        self.device = device

    # TODO: update with Paxton Frady's more recent resonator algorithm
    """
    TODO: visualization functionality
      - Image-space visualization (overlay view-deformation grid)
      - Template Canonical-coordinate visualization
      - VSA attribute indicator - show which attribute of which template is being actively factored out
      
      - Smooth transitions for visualizing factoring operators would be nice
      - Side project idea(s):
          - Compiler for automatically generating algorithm visualization code would be really nice
          - General set of torch visualization utils for gif, torch -> matplotlib figure creation would be great
          
    TODO: 
    """
    def decode(self, bound_vec, max_steps=100):
        x_states = []
        x_hists = []

        for iD in range(len(self.attr_dicts)):
            N = self.attr_dicts[iD].shape[1]
            Dv = self.attr_dicts[iD].shape[0]

            x_st = ru.cvec(N, 1).to(self.device).type(torch.complex64)
            x_st = x_st / complex_abs(x_st)
            x_states.append(x_st)

            x_hi = torch.zeros((max_steps, Dv)).to(self.device)
            x_hists.append(x_hi)

        for i in range(max_steps):
            th_vec = bound_vec.clone()
            all_converged = torch.zeros(len(self.attr_dicts), dtype=torch.bool)
            for iD in range(len(self.attr_dicts)):
                if i > 1:
                    xidx = torch.argmax(torch.abs(x_hists[iD][i - 1, :]))
                    x_states[iD] *= torch.sign(x_hists[iD][i - 1, xidx])

                th_vec *= torch.conj(x_states[iD])

            for iD in range(len(self.attr_dicts)):
                x_upd = th_vec / torch.conj(x_states[iD])

                mm = (torch.conj(self.attr_dicts[iD]) @ x_upd)
                mm.imag = torch.zeros_like(mm.imag)
                x_upd = self.attr_dicts[iD].T @ mm

                x_states[iD] = x_upd / complex_abs(x_upd)

                x_hists[iD][i, :] = torch.real(torch.conj(self.attr_dicts[iD]) @ x_states[iD])

                if i > 1:
                    all_converged[iD] = torch.allclose(x_hists[iD][i, :], x_hists[iD][i - 1, :], atol=5e-3, rtol=2e-2)

            # End if converged
            if torch.all(all_converged):
                print('converged:', i)
                break

        return x_hists, i

    # def simple_decode(self, bound_vec, max_steps=100):
    #     last_guess_vecs = None
    #     guess_vecs = ru.cvec(N, 3).to(self.device)
    #     # all_converged = torch.zeros(len(self.attr_dicts), dtype=torch.bool)
    #
    #     iters = 0
    #     while iters < 1 or not torch.allclose(last_guess_vecs, guess_vecs, atol=5e-3, rtol=2e-2):
    #         last_guess_vecs = guess_vecs
    #         for i, dict in enumerate(self.attr_dicts):
    #             guess_vecs[i] = dict.T @ torch.conj(dict) @ guess_vecs[i]
    #             # guess_vecs[i].imag = torch.zeros_like(guess_vecs[i])
    #             guess_vecs[i] /= torch.sqrt(torch.dot(torch.conj(guess_vecs[i]), guess_vecs[i]))
    #         iters += 1
    #
    #     for i, dict in enumerate(self.attr_dicts):
    #         best_idx = torch.argmax((torch.conj(dict) @ guess_vecs[i]).real)
    #         print(best_idx)
    #         guess_vecs[i] = dict[best_idx]
    #     print(torch.sum(bound_vec - guess_vecs[0] * guess_vecs[1] * guess_vecs[2]))
    #
    #     plt.imshow(vsa.decode_pix(guess_vecs[0] * guess_vecs[1] * guess_vecs[2]).cpu())
    #     plt.show()
    #     torch.conj(dicts_tensor) @ guess_vecs



def complex_abs(vec):
    """
    Computes vector of norms for each complex number in input
    :param vec:
    :return:
    """
    return torch.sqrt(torch.conj(vec) * vec)


def gen_pos_dicts(Vt, Ht, imshape):
    Vt_span = torch.zeros((imshape[0], N), dtype=torch.complex64)  # ru.crvec(N, Vspan)
    Ht_span = torch.zeros((imshape[1], N), dtype=torch.complex64)  # ru.crvec(N, Hspan)

    for i in range(imshape[0]):
        Vt_span[i, :] = Vt ** (i - imshape[0] // 2)

    for i in range(imshape[1]):
        Ht_span[i, :] = Ht ** (i - imshape[1] // 2)

    return [Vt_span, Ht_span]


def gen_image_vsa_vecs(letter_ims, vsa):
    reps = []
    for i, im in enumerate(letter_ims):
        f_vec = vsa.encode_pix(im)
        reps.append(f_vec)
    return torch.stack(reps).cuda()


def svd_whiten(X):
    U, s, V = torch.svd(X, compute_uv=True)
    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    return U @ V.T


if __name__ == "__main__":
    from letters import gen_letter_images
    from scae.util import vis

    import time


    N = int(3e4)
    patch_size = (56, 56)
    # These are special base vectors for position that loop
    # Vt = ru.cvec(N)  # , font_ims[0].shape[0])
    # Ht = ru.cvec(N)  # , font_ims[0].shape[1])
    Vt = ctvec(N, patch_size[1])
    Ht = ctvec(N, patch_size[0])
    # This is a set of 3 independently random complex phasor vectors for color
    # Cv = ru.crvec(N, 3)
    # Lv = ru.crvec(N, 26)


    with torch.no_grad():
        vsa = VSA(Vt, Ht, None, patch_size, device='cuda')

        letter_ims = gen_letter_images(patch_size)
        letter_dict = gen_image_vsa_vecs(letter_ims, vsa)
        letter_ims_w = svd_whiten(letter_ims.reshape(letter_ims.shape[0], -1)).reshape(letter_ims.shape)
        letter_dict_w = gen_image_vsa_vecs(letter_ims_w, vsa)
        # vis.plot_image_tensor(letter_ims_w.permute(0, 3, 1, 2))

        # bound_vec = letter_dict[0] * vsa.Vt ** 5 #+ reps[1] + reps[2] * vsa.Ht ** 9 + reps[3] * vsa.Vt ** 9 * vsa.Ht ** 9
        # bound_vec /= complex_abs(bound_vec)

        im_idx1 = np.random.randint(len(letter_ims))
        tH = patch_size[1] * np.random.rand(1)
        tV = patch_size[0] * np.random.rand(1)
        # rC = np.random.randint(colors_arr.shape[0])

        t_im1 = letter_ims[im_idx1].numpy()

        # for i in range(t_im1.shape[2]):
        #     t_im1[:, :, i] = colors_arr[rC, i] * t_im1[:, :, i]
        from scipy.ndimage.interpolation import shift
        t_im1 = shift(t_im1, (tV, tH, 0), mode='wrap', order=1)
        t_im = np.clip(t_im1, 0, 1)
        plt.imshow(t_im)
        plt.show()
        bound_vec = vsa.encode_pix(t_im)


        # Show image information stored in vsa vec
        bound_vec /= complex_abs(bound_vec)
        plt.imshow(vsa.decode_pix(bound_vec).cpu())
        plt.show()

        attr_dicts = [letter_dict_w] + gen_pos_dicts(vsa.Vt, vsa.Ht, patch_size)
        res = Resonator(attr_dicts)

        # Run torch resonator implementation
        tst = time.time()
        res_hist, nsteps = res.decode(bound_vec, 200)
        print(f"torch resonator took {time.time() - tst}s")

        plt.figure(figsize=(8, 3))
        ru.resplot_im([h.cpu() for h in res_hist], nsteps)  # , labels=res_xlabels, ticks=res_xticks)
        plt.tight_layout()
        plt.show()

        # Run numpy resonator implementation
        import resonator_numpy as rn
        tst = time.time()
        res_hist_np, nsteps_np = rn.res_decode_abs(bound_vec.cpu().numpy(), [a.cpu().numpy() for a in attr_dicts], 200)
        res_hist_np = [torch.as_tensor(r) for r in res_hist_np]
        print(f"numpy resonator took {time.time() - tst}s")

        plt.figure(figsize=(8, 3))
        ru.resplot_im([h.cpu() for h in res_hist_np], nsteps)  # , labels=res_xlabels, ticks=res_xticks)
        plt.tight_layout()
        plt.show()

        # Compute best guess vsa vec
        # guess_vec = torch.ones_like(bound_vec, dtype=torch.cfloat).cuda()
        # for attr_hist, attr_dict in zip(res_hist, attr_dicts):
        #     guess_vec *= attr_dict.cuda()[torch.argmax(attr_hist[nsteps])]
        # guess_vec /= complex_abs(guess_vec)
        # # Show image information
        # plt.imshow(vsa.decode_pix(guess_vec).cpu())
        # plt.show()
        pass