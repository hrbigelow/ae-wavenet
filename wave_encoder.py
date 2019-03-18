from torch import nn
import mfcc
import rfield as rf


class ConvReLURes(nn.Module):
    def __init__(self, n_in, n_out, filter_sz, stride=1, do_res=True):
        self.do_res = do_res
        if self.do_res:
            if stride != 1:
                print('Stride must be 1 for residually connected convolution',
                        file=sys.stderr)
                raise ValueError

        self.n_in = n_in
        self.n_out = n_out
        self.conv = nn.Conv1d(n_in, n_out, filter_sz, stride, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.foff = rf.FieldOffset(filter_sz=filter_sz)

    def forward(self, x):
        '''
        B, C, T = n_batch, n_in, n_win
        x: (B, C, T)
        '''
        out = self.conv(x)
        out = self.relu(out)
        if (self.do_res):
            # Must suitably trim the residual based on how much the convolution
            # shrinks the input.
            # assert self.n_in == self.n_out
            out += x[self.foff.left:-self.foff.right]
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

        left_off = self.pre.left + sum(m.left for m in self.net.children())
        right_off = self.pre.right + sum(m.right for m in self.net.children())
        self.foff = rf.FieldOffset(offsets=(left_off, right_off))

    def forward(self, wav):
        '''
        B, T = n_batch, n_win + rf_size - 1
        wav: (B, T)
        '''
        mfcc = self.pre.func(wav)
        out = self.net(mfcc)
        return out

