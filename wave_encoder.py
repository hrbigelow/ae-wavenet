from torch import nn
import mfcc


class ConvReLURes(nn.Module):
    def __init__(self, n_in, n_out, n_kern, stride=1, do_res=True):
        super(ConvReLURes, self).__init__()
        self.do_res = do_res
        self.n_in = n_in
        self.n_out = n_out
        self.conv = nn.Conv1d(n_in, n_out, n_kern, stride, padding=int(n_kern/2))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        '''
        B, C, T = n_batch, n_in, n_win
        x: (B, C, T)
        '''
        out = self.conv(x)
        out = self.relu(out)
        if (self.do_res):
            # Residual connection only works if in and out dimensions are equal
            assert self.n_in == self.n_out
            out += x
        return out

class FCRes(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(FCRes, self).__init__()
        self.fc = nn.Linear(n_in, n_out, bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        '''
        B, T, C = n_batch, n_win, n_in
        x: (B, T, C)
        '''
        out = self.fc(x)
        out = self.relu(out)
        out += x
        return out


class Encoder(nn.Module):
    def __init__(self, sample_rate_ms, win_length_ms, hop_length_ms,
            n_mels, n_mfcc, n_out):
        super(Encoder, self).__init__()

        self.pre = mfcc.ProcessWav(
                sample_rate_ms, win_length_ms, hop_length_ms, n_mels, n_mfcc)

        n_in = self.pre.n_out
        
        self.net = nn.Sequential(
            ConvReLURes(n_in, n_out, 3, do_res=False),
            ConvReLURes(n_out, n_out, 3),
            ConvReLURes(n_out, n_out, 4, stride=2, do_res=False),
            ConvReLURes(n_out, n_out, 3),
            ConvReLURes(n_out, n_out, 3),
            ConvReLURes(n_out, n_out, 1),
            ConvReLURes(n_out, n_out, 1),
            ConvReLURes(n_out, n_out, 1),
            ConvReLURes(n_out, n_out, 1)
        )

    def forward(self, wav):
        '''
        B, T = n_batch, n_win + rf_size - 1
        wav: (B, T)
        '''
        mfcc = self.pre.func(wav)
        out = self.net(mfcc)
        return out

