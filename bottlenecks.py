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
        self.dist = dist.MultivariateNormal(torch.zeros(n_out), torch.eye(n_out)) 

    def forward(x):
        out = self.fc(x)
        self.dist.mean[:] = out[0:n_out]
        self.dist.covariance_matrix[:,:] = torch.diag(out[n_out:])
        out = self.dist.sample()
        return out


class AE(nn.Module):
    def __init__(self, n_in, otu_channels, bias):
        self.fc = nn.Linear(n_in, n_out, bias)

    def forward(x):
        out = self.fc(x)
        return out


