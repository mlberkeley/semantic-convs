import torch
from PIL import ImageFont


class Letters(torch.utils.data.Dataset):
    def __init__(self, num_letters=26, transform=None):
        self.num_letters = num_letters
        self.transform = transform
        self.images = self.gen_letter_images()[:num_letters]


    def gen_letter_images(self, patch_size = (56, 56)):
        font = ImageFont.truetype(u'/usr/share/fonts/opentype/mathjax/MathJax_Typewriter-Regular.otf', size=18)
        letters = 'abcdefghijklmnopqrstuvwxyz'
        font_ims = []

        fim_size = (patch_size[0], patch_size[1], 1)

        for l in letters:
            font_obj = font.getmask(l)

            imtext = torch.as_tensor(font_obj)
            imsize = font_obj.size  # font.getsize(l)

            imtext = imtext.reshape((imsize[1], imsize[0], 1))
            imtext = imtext[:patch_size[0], :patch_size[1], :] / 255
            font_ims.append(imtext)

            # imsize = imtext.shape
            #
            # fim = np.zeros(fim_size)
            # fimr = int(np.floor((fim_size[0] - imsize[0]) / 2))
            # fimc = int(np.floor((fim_size[1] - imsize[1]) / 2))
            # fim[fimr:(fimr + imsize[0]), fimc:(fimc + imsize[1]), :] = imtext
            #
            # font_ims.append(fim)
        # t_font_ims = torch.Tensor(font_ims).permute(0, 3, 1, 2)[:, 0:1]

        # max_size = [0, 0]
        max_size = [max(font_ims, key=lambda i: i.shape[0]).shape[0],
                    max(font_ims, key=lambda i: i.shape[1]).shape[1]]
        t_font_ims = torch.zeros(len(letters), 1, *max_size)
        for idx, im in enumerate(font_ims):
            t_font_ims[idx, :, :im.shape[0], :im.shape[1]] = im[..., 0]

        # for l in font_ims:
        #     if l.size[0] > max_size[0]:
        #         max_size[0] = l.size[0]
        #     if l.size[1] > max_size[1]:
        #         max_size[1] = l.size[1]


        return t_font_ims  # shape = num_ims, channel, width, height

    def __getitem__(self, item):
        """
        Randomly inserts the MNIST images into cifar images
        :param item:
        :return:
        """
        idx = item
        image = self._norm_img(self.images[idx])
        return self.transform(image) if self.transform else image

    def __len__(self):
        return len(self.images)

    @staticmethod
    def _norm_img(image):
        image = torch.abs(image - image.quantile(.5))
        i_max = torch.max(image)
        i_min = torch.min(image)
        image = torch.div(image - i_min, i_max - i_min + 1e-8)
        return image

if __name__ == "__main__":
    d = Letters()
    from scae.util.plots import plot_image_tensor
    plt, _ = plot_image_tensor(d.images)
    plt.show()

