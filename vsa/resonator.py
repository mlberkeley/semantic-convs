import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F

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
            x_st = x_st / ru.complex_abs(x_st)
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

                x_states[iD] = x_upd / ru.complex_abs(x_upd)

                x_hists[iD][i, :] = torch.real(torch.conj(self.attr_dicts[iD]) @ x_states[iD])

                if i > 1:
                    all_converged[iD] = torch.allclose(x_hists[iD][i, :], x_hists[iD][i - 1, :], atol=5e-3, rtol=2e-2)

            # End if converged
            if torch.all(all_converged):
                # print('converged:', i)
                break

        return x_hists, i


def gen_pos_dicts(Vt, Ht, imshape, template_centered=True):
    Vt_span = torch.zeros((imshape[0], Vt.shape[-1]), dtype=torch.complex64)  # ru.crvec(N, Vspan)
    Ht_span = torch.zeros((imshape[1], Ht.shape[-1]), dtype=torch.complex64)  # ru.crvec(N, Hspan)

    for i in range(imshape[0]):
        idx = i - imshape[0] // 2 if template_centered else i
        Vt_span[i, :] = Vt ** idx

    for i in range(imshape[1]):
        idx = i - imshape[1] // 2 if template_centered else i
        Ht_span[i, :] = Ht ** idx

    return [Vt_span, Ht_span]


def gen_image_vsa_vecs(letter_ims, vsa):
    reps = []
    for i, im in enumerate(letter_ims):
        f_vec = vsa.encode_pix(im)
        reps.append(f_vec)
    return torch.stack(reps).cuda()


if __name__ == "__main__":

    from scae.util import vis
    import time

    # dataset = "letters"
    dataset = "mnist_objects"
    run_numpy = False

    N = int(3e4)
    if dataset == "letters":
        template_size = (56, 56)
        image_size = (56, 56)
        template_centered = True
        from letters import gen_letter_images
        template_ims = gen_letter_images(template_size)
    elif dataset == "mnist_objects":
        template_size = (11, 11)
        image_size = (40, 40)
        template_centered = True
        # from torch.utils.data import DataLoader
        from scae.data.mnist_objects import MNISTObjects
        dataset = MNISTObjects(train=True)
        # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        template_ims = dataset.templates
    else:
        raise NotImplementedError(f"No dataset {dataset}")

    # These are special base vectors for position that loop
    # Vt = ru.cvec(N)  # , font_ims[0].shape[0])
    # Ht = ru.cvec(N)  # , font_ims[0].shape[1])
    Vt = ctvec(N, image_size[1])
    Ht = ctvec(N, image_size[0])
    # This is a set of 3 independently random complex phasor vectors for color
    # Cv = ru.crvec(N, 3)
    # Lv = ru.crvec(N, 26)

    with torch.no_grad():
        # Init VSA encoder / decoder
        vsa = VSA(Vt, Ht, None, image_size)

        # Init Resonator
        template_dict = gen_image_vsa_vecs(template_ims, vsa)
        template_ims_w = ru.svd_whiten(template_ims)
        template_dict_w = gen_image_vsa_vecs(template_ims_w, vsa)
        vis.plot_image_tensor(
            torch.cat([template_ims, template_ims_w], dim=0)
        )

        res_attr_dicts = [template_dict_w] + gen_pos_dicts(vsa.Vt, vsa.Ht, image_size, template_centered)
        gt_attr_dicts = [template_dict] + res_attr_dicts[1:]
        res = Resonator(res_attr_dicts)

        # Evaluation Loop
        batch_outs = []
        res_total_time = 0.
        num_batches = 20
        for i in range(num_batches):
            # bound_vec = letter_dict[0] * vsa.Vt ** 5 #+ reps[1] + reps[2] * vsa.Ht ** 9 + reps[3] * vsa.Vt ** 9 * vsa.Ht ** 9
            # bound_vec /= complex_abs(bound_vec)
            pad_amt = [0, image_size[0] - template_size[0], 0, image_size[1] - template_size[1]]

            im_idx1 = np.random.randint(len(template_ims))
            tH = pad_amt[1] * np.random.rand(1)
            tV = pad_amt[3] * np.random.rand(1)
            # rC = np.random.randint(colors_arr.shape[0])
            t_im1 = F.pad(template_ims[im_idx1], pad_amt).cpu().numpy()

            # for i in range(t_im1.shape[2]):
            #     t_im1[:, :, i] = colors_arr[rC, i] * t_im1[:, :, i]
            from scipy.ndimage.interpolation import shift
            t_im1 = shift(t_im1, (tV, tH, 0), mode='wrap', order=1)
            t_im = np.clip(t_im1, 0, 1)
            bound_vec = vsa.encode_pix(t_im)

            if run_numpy:
                # Run numpy resonator implementation
                import resonator_numpy as rn
                tst = time.time()
                res_hist_np, nsteps_np = rn.res_decode_abs(bound_vec.cpu().numpy(),
                                                           [a.cpu().numpy() for a in res_attr_dicts], 200)
                res_hist_np = [torch.as_tensor(r) for r in res_hist_np]
                print(f"numpy resonator took {time.time() - tst}s")

                # Plot convergence dynamics
                plt.figure(figsize=(8, 3))
                ru.resplot_im([h.cpu() for h in res_hist_np], nsteps)  # , labels=res_xlabels, ticks=res_xticks)
                plt.tight_layout()
                plt.show()

            # Run torch resonator implementation
            tst = time.time()
            res_hist, nsteps = res.decode(bound_vec, 200)
            res_total_time += time.time() - tst

            # Plot convergence dynamics
            # plt.figure(figsize=(8, 3))
            # ru.resplot_im([h.cpu() for h in res_hist], nsteps)  # , labels=res_xlabels, ticks=res_xticks)
            # plt.tight_layout()
            # plt.show()

            # Compute best guess vsa vec
            guess_vec = torch.ones_like(bound_vec, dtype=torch.complex64).cuda()
            for attr_hist, attr_dict in zip(res_hist, gt_attr_dicts):
                guess_vec *= attr_dict.cuda()[torch.argmax(attr_hist[nsteps])]

            # Show input image, image decoded from vsa vec (encoded img), and image decoded from guess vec
            batch_outs.append(
                torch.stack([
                    torch.as_tensor(t_im),
                    # vsa.decode_pix(bound_vec).permute(2, 0, 1).cpu(),
                    vsa.decode_pix(guess_vec).permute(2, 0, 1).cpu()
                ], dim=0)
            )

        print(f"torch resonator took avg of {res_total_time / num_batches}s")
        vis.plot_image_tensor_2D(
            torch.stack(batch_outs, dim=0), # shape = rows, columns, image
            titles=[
                "Input\nimage",
                # "Encoded\nimage",
                "Decoded\nimage"
            ]
        )
        pass