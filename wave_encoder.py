import torch
from torch import nn
import mfcc
import rfield
import numpy as np


class ConvReLURes(nn.Module):
    def __init__(self, n_in_chan, n_out_chan, filter_sz, stride=1, do_res=True,
            parent_rf=None, name=None):
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

        self.rf = rfield.Rfield(filter_info=filter_sz, stride=stride,
                parent=parent_rf, name=name)

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
            __, l_off, r_off, __ = self.rf.geometry(10)
            out += x[:,:,l_off:r_off or None]
        return out


class Encoder(nn.Module):
    def __init__(self, sample_rate, win_sz, hop_sz, n_mels, n_mfcc, n_out):
        super(Encoder, self).__init__()

        # the "preprocessing"
        self.pre = mfcc.ProcessWav(sample_rate, win_sz, hop_sz, n_mels, n_mfcc, 'mfcc')

        # the "stack"
        stack_in_chan = [self.pre.n_out, n_out, n_out, n_out, n_out, n_out, n_out, n_out, n_out]
        stack_filter_sz = [3, 3, 4, 3, 3, 1, 1, 1, 1]
        stack_strides = [1, 1, 2, 1, 1, 1, 1, 1, 1]
        stack_residual = [False, True, False, True, True, True, True, True, True]
        stack_info = zip(stack_in_chan, stack_filter_sz, stack_strides, stack_residual)

        self.net = nn.Sequential()
        parent_rf = self.pre.rf

        for i, (in_chan, filt_sz, stride, do_res) in enumerate(stack_info):
            name = 'CRR_{}(filter_sz={}, stride={}, do_res={})'.format(i,
                    filt_sz, stride, do_res)
            mod = ConvReLURes(in_chan, n_out, filt_sz, stride, do_res,
                    parent_rf, name)
            self.net.add_module(str(i), mod)
            parent_rf = mod.rf

        self.rf = parent_rf 

    def forward(self, wav):
        '''
        B, T = n_batch, n_win + rf_size - 1
        wav: (B, T) (torch.tensor)
        '''
        # Note: for consistency, we would like all 'forward' calls to accept
        # and return a torch.tensor.  This one happens to require
        # pre-processing functions that only work on numpy.ndarrays, so first
        # convert to numpy, and then back.
        mels = torch.tensor(np.apply_along_axis(self.pre.func, axis=1, arr=wav),
                dtype=torch.float)
        out = self.net(mels)
        return out

