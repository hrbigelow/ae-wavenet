from torch import nn
from torch.distributions import multivariate_normal as dist

class VQVAE(nn.Module):
    def __init__(self, n_in, n_out, bias):
        self.proto_vec = 0
        self.fc = nn.Linear(n_in, n_out, bias)
        self.nearest_proto()

    def forward(x):
        out = self.fc(x)
        out = self.nearest_proto(out)
        return out


class VAE(nn.Module):
    def __init__(self, n_in, n_out, bias):
        self.fc = nn.Linear(n_in, n_out * 2, bias)
        # dummy dimensions - will be set in forward
        self.dist = dist.Normal(torch.tensor([0]), torch.tensor([0]))

    def forward(x):
        # Input: (N, T, I)
        out = self.fc(x)
        self.dist.loc = out[:,:,:n_out]
        self.dist.scale = out[:,:,n_out:]
        out = self.dist.rsample([1]).squeeze(0)
        return out


class AE(nn.Module):
    def __init__(self, n_in, n_out, bias):
        self.fc = nn.Linear(n_in, n_out, bias)

    def forward(x):
        out = self.fc(x)
        return out


