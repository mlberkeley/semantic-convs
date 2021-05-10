import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from scipy.ndimage.interpolation import shift

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
    num_templates = 3

    N = int(3e4)
    if dataset == "letters":
        template_centered = True
        from letters import gen_letter_images
        template_ims = gen_letter_images()

        template_size = template_ims[0].shape[1:]
        image_size = (40, 40)
    elif dataset == "mnist_objects":

        template_centered = True
        # from torch.utils.data import DataLoader
        from scae.data.mnist_objects import MNISTObjects
        dataset = MNISTObjects(train=True)
        # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        template_ims = dataset.templates

        template_size = template_ims[0].shape[1:]
        image_size = (40, 40)
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
        num_batches = 10
        for i in range(num_batches):
            # Datapoint generation
            # bound_vec = letter_dict[0] * vsa.Vt ** 5 #+ reps[1] + reps[2] * vsa.Ht ** 9 + reps[3] * vsa.Vt ** 9 * vsa.Ht ** 9

            pad_amt = [0, image_size[1] - template_size[1], 0, image_size[0] - template_size[0]]
            target_image = np.zeros(image_size)[None, ...]
            for t in range(num_templates):
                template_idx = np.random.randint(len(template_ims))
                tH = pad_amt[3] * np.random.rand(1)
                tV = pad_amt[1] * np.random.rand(1)
                # rC = np.random.randint(colors_arr.shape[0])
                template = F.pad(template_ims[template_idx], pad_amt).cpu().numpy()

                # for i in range(t_im1.shape[2]):
                #     t_im1[:, :, i] = colors_arr[rC, i] * t_im1[:, :, i]
                template = shift(template, (0, tV, tH), mode='wrap', order=1)
                target_image += template
            target_image = np.clip(target_image, 0, 1)

            # TODO: end on high cossim or or low residual (bound_vec, guess_vec)
            bound_vecs = [vsa.encode_pix(target_image)]
            guess_agg_vecs = []
            for t in range(num_templates):
                # Run torch resonator implementation
                tst = time.time()
                res_hist, nsteps = res.decode(bound_vecs[-1], 200)
                res_total_time += time.time() - tst

                # Plot convergence dynamics
                # plt.figure(figsize=(8, 3))
                # ru.resplot_im([h.cpu() for h in res_hist], nsteps)  # , labels=res_xlabels, ticks=res_xticks)
                # plt.tight_layout()
                # plt.show()

                # Compute best guess vsa vec
                guess_vec = torch.ones_like(bound_vecs[-1], dtype=torch.complex64).cuda()
                for attr_hist, attr_dict in zip(res_hist, gt_attr_dicts):
                    guess_vec *= attr_dict.cuda()[torch.argmax(attr_hist[nsteps])]
                guess_agg_vecs.append(guess_agg_vecs[-1] + guess_vec if len(guess_agg_vecs) > 0 else guess_vec)

                residual_vec = bound_vecs[-1] - guess_vec
                bound_vecs.append(residual_vec)

            # Show input image, image decoded from vsa vec (encoded img), and image decoded from guess vec
            target_image_row = [vsa.decode_pix(b_vec).permute(2, 0, 1).cpu() for b_vec in bound_vecs]
            guess_image_row = [vsa.decode_pix(g_vec).permute(2, 0, 1).cpu() for g_vec in guess_agg_vecs]
            # Compute and add diff image between input and recon

            # Append grid
            batch_outs.append(
                torch.stack(
                    target_image_row
                    + guess_image_row
                    + [torch.abs(target_image_row[0] - guess_image_row[-1])],
                    dim=0)
            )

        print(f"torch resonator took avg of {res_total_time / num_batches}s per decode")
        sims = torch.as_tensor(
            [bo[-1].sum().item() for bo in batch_outs]
        ) # L1 reconstruction diff (sum over all pixels over difference image)
        batch_outs = torch.stack(batch_outs, dim=0)
        _, indices = torch.sort(sims, descending=True)

        vis.plot_image_tensor_2D(
            batch_outs[indices], # shape = rows, columns, image
            titles=["Target"]
                   + [f"{i} Out" for i in range(1, num_templates+1)]
                   + [f"{i} In" for i in range(1, num_templates+1)]
                   + ["Error"]
        )
        pass