from torch import nn

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
        self.sample = gaussian_sample()


    def forward(x):
        out = self.fc(x)
        out = self.sample(out[0:n_out],
                out[n_out:n_out * 2])
        return out


class AE(nn.Module):
    def __init__(self, n_in, otu_channels, bias):
        self.fc = nn.Linear(n_in, n_out, bias)

    def forward(x):
        out = self.fc(x)
        return out


