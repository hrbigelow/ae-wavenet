from torch import nn
import netmisc

class AE(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(AE, self).__init__()
        self.linear = nn.Conv1d(n_in, n_out, 1, bias=bias)
        netmisc._xavier_init(self.linear)

    def forward(self, x):
        out = self.linear(x)
        return out

