class ConvRes(nn.Module):
    def __init__(self, n_in, n_out, n_kern, stride=1):
        super(ConvRes, self).__init__()
        self.conv = nn.Conv1d(n_in, n_out, n_kern, stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
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
        self.net = nn.Sequential(
            ConvRes(n_in, n_mid, 3),
            ConvRes(n_mid, n_mid, 3),
            ConvRes(n_mid, n_mid, 4, stride=2),
            ConvRes(n_mid, n_mid, 3),
            ConvRes(n_mid, n_mid, 3),
            FCRes(n_mid, n_mid),
            FCRes(n_mid, n_mid),
            FCRes(n_mid, n_mid),
            FCRes(n_mid, n_mid)
            )

    def forward(x):
        out = self.net(x)
        return out
