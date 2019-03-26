import torch
from torch import nn
import mfcc
import rfield as rf
import numpy as np



class ConvReLURes(nn.Module):
    def __init__(self, n_in_chan, n_out_chan, filter_sz, stride=1, do_res=True):
        super(ConvReLURes, self).__init__()
        self.do_res = do_res
        if self.do_res:
            if stride != 1:
                print('Stride must be 1 for residually connected convolution',
                        file=sys.stderr)
                raise ValueError

        self.n_in = n_in_chan
        self.n_out = n_out_chan
        self.conv = nn.Conv1d(n_in_chan, n_out_chan, filter_sz, stride, padding=0)
        self.relu = nn.ReLU(inplace=True)

        self.foff = rf.FieldOffset(filter_sz=filter_sz)

    def forward(self, x):
        '''
        B, C, T = n_batch, n_in_chan, n_win
        x: (B, C, T)
        '''
        out = self.conv(x)
        out = self.relu(out)
        if (self.do_res):
            # Must suitably trim the residual based on how much the convolution
            # shrinks the input.
            # assert self.n_in == self.n_out
            out += x[:,:,self.foff.left:-self.foff.right or None]
        return out


class Encoder(nn.Module):
    def __init__(self, sample_rate_ms, win_length_ms, hop_length_ms,
            n_mels, n_mfcc, n_out):
        super(Encoder, self).__init__()

        # the "preprocessing"
        self.pre = mfcc.ProcessWav(
                sample_rate_ms, win_length_ms, hop_length_ms, n_mels, n_mfcc)

        
        # the "stack"
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

        # the offsets in stack element coordinates
        stack_loff = sum(m.foff.left for m in self.net.children())
        stack_roff = sum(m.foff.right for m in self.net.children())

        # spacing of the input of the stack in timesteps 
        input_stepsize = sample_rate_ms * hop_length_ms
        left_off = self.pre.foff.left + stack_loff * input_stepsize
        right_off = self.pre.foff.right + stack_roff * input_stepsize
        self.foff = rf.FieldOffset(offsets=(left_off, right_off))


    def forward(self, wav):
        '''
        B, T = n_batch, n_win + rf_size - 1
        wav: (B, T) (torch.tensor)
        '''
        # Note: for consistency, we would like all 'forward' calls to accept
        # and return a torch.tensor.  This one happens to require
        # pre-processing functions that only work on numpy.ndarrays, so first
        # convert to numpy, and then back.
        mels = torch.tensor(np.apply_along_axis(self.pre.func, axis=1, arr=wav.numpy()),
                dtype=torch.float)
        out = self.net(mels)
        return out

