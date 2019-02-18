from torch import nn

class ConvReLURes(nn.Module):
    def __init__(self, n_in, n_out, n_kern, stride=1, do_res=True):
        super(ConvReLURes, self).__init__()
        self.do_res = do_res
        self.conv = nn.Conv1d(n_in, n_out, n_kern, stride, padding=int(n_kern/2))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        if (self.do_res):
            out += x
        return out

class FCRes(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(FCRes, self).__init__()
        self.fc = nn.Linear(n_in, n_out, bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        out += x
        return out

class Encoder(nn.Module):
    def __init__(self, n_in, n_mid):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            ConvReLURes(n_in, n_mid, 3, do_res=False),
            ConvReLURes(n_mid, n_mid, 3),
            ConvReLURes(n_mid, n_mid, 4, stride=2, do_res=False),
            ConvReLURes(n_mid, n_mid, 3),
            ConvReLURes(n_mid, n_mid, 3),
            ConvReLURes(n_mid, n_mid, 1),
            ConvReLURes(n_mid, n_mid, 1),
            ConvReLURes(n_mid, n_mid, 1),
            ConvReLURes(n_mid, n_mid, 1)
            )

    def forward(self, x):
        out = self.net(x)
        return out
