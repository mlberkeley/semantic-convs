from scae.modules.flow_decoder import ImplicitDecoder
from scae.modules.flow_encoder import CapsuleEncoder
import torch

class FCAE(torch.nn.Module):
    def __init__(self, encoder, decoder, num_caps=16, input_dims=2, latent_dims=8, transform_dims=4):
        super(FCAE, self).__init__()
        self.caps_dims = input_dims + latent_dims + transform_dims
        self.encoder = CapsuleEncoder(num_caps, self.caps_dims)
        self.decoder = ImplicitDecoder(latent_dims)

    def forward(self, im1, im2):
        o1 = self.encoder(im1)
        caps_set_1 = self.encoder.gen_capsule_set(o1)
        o2 = self.encoder(im2)
        caps_set_2 = self.encoder.gen_capsule_set(o2)


    def create_capsule_flow(self, set1, set2):
        pass
