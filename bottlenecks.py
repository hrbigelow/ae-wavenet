from torch import nn
from torch.distributions import multivariate_normal as dist

class VQVAE(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(VQVAE, self).__init__()
        self.proto_vec = 0
        self.linear = nn.Conv1d(n_in, n_out, 1, bias=bias)
        self.nearest_proto()

    def forward(self, x):
        out = self.linear(x)
        out = self.nearest_proto(out)
        return out


class VAE(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(VAE, self).__init__()
        self.linear = nn.Conv1d(n_in, n_out * 2, 1, bias=bias)
        # dummy dimensions - will be set in forward
        self.dist = dist.Normal(torch.tensor([0]), torch.tensor([0]))

    def forward(self, x):
        # Input: (N, T, I)
        out = self.linear(x)
        self.dist.loc = out[:,:,:n_out]
        self.dist.scale = out[:,:,n_out:]
        out = self.dist.rsample([1]).squeeze(0)
        return out


class AE(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(AE, self).__init__()
        self.linear = nn.Conv1d(n_in, n_out, 1, bias=bias)

    def forward(self, x):
        out = self.linear(x)
        return out

