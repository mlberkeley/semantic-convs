import matplotlib.pyplot as plt
import numpy as np

import torch

import notebooks.res_utils_frady
import res_utils_torch as ru



class VSA:
    def __init__(self, Vt, Ht, Cv, imshape, device="cuda"):
        self.Vt = torch.as_tensor(Vt, dtype=torch.complex64).to(device)
        self.Ht = torch.as_tensor(Ht, dtype=torch.complex64).to(device)
        # self.Cv = torch.as_tensor(Cv, dtype=torch.complex128).to(device)

        # VSA vec length
        self.V = self.Vt.shape[-1]
        Hr = torch.arange(0, imshape[0]).to(device)[:, None, None, None].expand(imshape[0], 1, 1, self.V)
        Vr = torch.arange(0, imshape[1]).to(device)[None, :, None, None].expand(1, imshape[1], 1, self.V)

        self.P_vec = self.Ht[None, None, None, :].expand(1, imshape[1], 1, self.V).pow(Hr) \
                     * self.Vt[None, None, None, :].expand(imshape[0], 1, 1, self.V).pow(Vr) \
                     # * self.Cv[None, None, :, :]

        # Position "hashing" vectors (Verticals, Horizontals, Channels, VSA vec lengths)
        print(f"P_vec shape: {self.P_vec.shape}")
        self.device = device

    def encode_pix(self, im):
        if not isinstance(im, torch.Tensor):
            im = torch.as_tensor(im, dtype=torch.float)
        if im.shape[0] <= 3:
            im = im.permute(1, 2, 0)
        im = im.to(self.device)
        # Weighted sum of all P_vectors over pixels and channels (superposition of pixel values, embedded with P_vec)
        return (self.P_vec.T * im.T).reshape(self.V, -1).sum(-1).T

    def decode_pix(self, image_vec):
        if not isinstance(image_vec, torch.Tensor):
            image_vec = image_vec.as_tensor(image_vec, dtype=torch.complex64).to(self.device)
        image_vec = image_vec.to(self.device)
        # Selects pixel value by dotting VSA vec with conjugate of position encoding, and taking real component (clip for stability)
        # Old elementwise formula: return_img[m, n, c] = torch.real(torch.dot(torch.conj(self.P_vec[m, n, c]), image_vec) / self.R)
        return torch.clip(torch.real(torch.conj(self.P_vec) * image_vec).sum(-1) / self.V, 0, 1)


def ctvec(N, loop):
    # randomly samples complex vector for toroidal embedding with cycle of size "loop"
    return torch.as_tensor(notebooks.res_utils_frady.cvecl(N, loop), dtype=torch.complex64)
    # torch.exp(2j * np.pi * torch.randint(loop, (N,)) / loop)

# with torch.no_grad():
#     vsa = VSA(Vt, Ht, Cv)
#     bound_vec = vsa.encode_pix(t_im).cpu()
#     recovered_im = vsa.decode_pix(bound_vec).cpu()
#     plt.imshow(recovered_im)
#     plt.show()
#
#     del vsa

if __name__ == "__main__":
    from letters import gen_letter_images
    from scae.util import vis
    import time

    N = int(3e4)
    # These are special base vectors for position that loop
    Vt = ru.cvec(N)  # , font_ims[0].shape[0])
    Ht = ru.cvec(N)  # , font_ims[0].shape[1])
    # Vt = ctvec(N, patch_size[1])
    # Ht = ctvec(N, patch_size[0])
    # This is a set of 3 independently random complex phasor vectors for color
    Cv = ru.crvec(N, 3)
    patch_size = (56, 56)

    font_ims = gen_letter_images(patch_size)
    with torch.no_grad():
        vsa = VSA(Vt, Ht, Cv, patch_size, device='cuda')

        total = len(font_ims)
        total_encode = 0.
        total_decode = 0.
        ims = []
        reps = []
        recs = []

        for i, im in enumerate(font_ims):
            ims.append(im)

            tst = time.time()
            f_vec = vsa.encode_pix(im)
            total_encode += time.time() - tst
            reps.append(f_vec)

            # translate the image
            f_vec_tr = f_vec * vsa.Vt ** i * vsa.Ht ** i

            tst = time.time()
            f_im = vsa.decode_pix(f_vec_tr)
            total_decode += time.time() - tst
            recs.append(f_im.cpu())

        f_vec_composite = reps[0] * vsa.Vt ** -9 + reps[1] + reps[2] * vsa.Ht ** 9 + reps[3] * vsa.Vt ** 9 * vsa.Ht ** 9
        tst = time.time()
        f_im = vsa.decode_pix(f_vec_composite)
        print(f"composite decode time: {time.time() - tst:.3}s")
        plt.imshow(f_im.cpu())
        plt.show()

        print(f"avg encode time: {total_encode / total:.3}s")
        print(f"avg decode time: {total_decode / total:.3}s")
        vis.plot_image_tensor(torch.stack(ims).permute(0, 3, 1, 2))
        vis.plot_image_tensor(torch.stack(recs).permute(0, 3, 1, 2))

        del vsa
        torch.cuda.empty_cache()
    #     print(torch.cuda.memory_stats(device=None))

