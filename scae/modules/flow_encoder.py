import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

class CapsuleEncoder(nn.Module):
    def __init__(self, num_caps, caps_dim, output_channels=[32,32,64,64], pool_dim=2):
        super(CapsuleEncoder, self).__init__()
        self.num_caps = num_caps
        self.caps_dim = caps_dim
        self.cnn = torch.nn.Sequential()
        self.fc = torch.nn.Sequential()
        for i in range(len(output_channels)):
            s = output_channels[i]
            if i == 0:
                self.cnn.add_module("conv_" + str(i), torch.nn.Conv2d(3, s, kernel_size=5))
            else:
                self.cnn.add_module("conv_" + str(i), torch.nn.Conv2d(output_channels[i-1],
                                                                      s, kernel_size=5))
            self.cnn.add_module("maxpool_" + str(i), torch.nn.MaxPool2d(kernel_size=2))
            self.cnn.add_module("relu_" + str(i), torch.nn.ReLU())

        # TODO: ensure correct shapes for FC layer
        self.fc.add_module("fc_0", torch.nn.Linear(output_channels[-1], 32))
        self.fc.add_module("tanh_0", torch.nn.Tanh())

        # TODO: a shared FC layer to get K capsules; but check if there should be K FC's.
        self.fc.add_module("fc_1", torch.nn.Linear(32, num_caps*caps_dim, bias=True))

    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x)

    def gen_capsule_set(self, x):
        return torch.split(x, self.caps_dim, dim=1)

    def conformal_map(self, theta):
        cmap = torch.Tensor([[theta[3] * torch.cos(theta[2]), -1 * theta[3] * torch.sin(theta[2]), theta[0]],
                            [theta[3] * torch.sin(theta[2]), theta[3] * torch.cos(theta[2]), theta[1]],
                            [0, 0, 1]])
        return cmap