from scae.modules.flow_decoder import ImplicitDecoder
from scae.modules.flow_encoder import CapsuleEncoder
import torch
# TODO: MNIST OBJECTS - util vis will be useful
class FCAE(torch.nn.Module):
    def __init__(self, encoder, decoder, num_caps=16, input_dims=2, latent_dims=8, transform_dims=4):
        super(FCAE, self).__init__()
        self.caps_dims = input_dims + latent_dims + transform_dims
        self.encoder = CapsuleEncoder(num_caps, self.caps_dims)
        self.decoder = ImplicitDecoder(latent_dims)
        self.num_caps = num_caps
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.tranform_dims = transform_dims

    def forward(self, im1, im2):
        batch_size = im.shape[0]
        h = im.shape[1]
        w = im.shape[2]
        decoder_input_shape = (batch_size, self.num_caps, (self.latent_dims + self.input_dims), h*w)
        flattened_decoder_input_shape = (decoder_input_shape[0]*decoder_input_shape[1], *decoder_input_shape[2:])

        o1 = self.encoder(im1)
        caps_set_1 = self.encoder.gen_capsule_set(o1)
        o2 = self.encoder(im2)
        caps_set_2 = self.encoder.gen_capsule_set(o2)

        dc = caps_set_1[:, :, :self.latent_dims].unsqueeze(-1).repeat(1, 1, 1, h*w)

        # we want batch x num_capsules x shape_vector_length + 2 x H*W

    def create_capsule_flow(self, set1, set2):
        pass
