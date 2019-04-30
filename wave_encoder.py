import torch
from torch import nn
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
        #self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()

        self.rf = rfield.Rfield(filter_info=filter_sz, stride=stride,
                parent=parent_rf, name=name)

    def forward(self, x):
        '''
        B, C, T = n_batch, n_in_chan, n_win
        x: (B, C, T)
        '''
        assert self.rf.src.nv == x.shape[2]
        out = self.conv(x)
        out = self.relu(out)
        if (self.do_res):
            l_off, r_off = rfield.offsets(self.rf, self.rf)
            out += x[:,:,l_off:r_off or None]
        assert self.rf.dst.nv == out.shape[2]
        return out


class Encoder(nn.Module):
    def __init__(self, n_in, n_out, parent_rf):
        super(Encoder, self).__init__()

        # the "stack"
        stack_in_chan = [n_in, n_out, n_out, n_out, n_out, n_out, n_out, n_out, n_out]
        stack_filter_sz = [3, 3, 4, 3, 3, 1, 1, 1, 1]
        stack_strides = [1, 1, 2, 1, 1, 1, 1, 1, 1]
        stack_residual = [False, True, False, True, True, True, True, True, True]
        stack_info = zip(stack_in_chan, stack_filter_sz, stack_strides, stack_residual)

        self.net = nn.Sequential()

        for i, (in_chan, filt_sz, stride, do_res) in enumerate(stack_info):
            name = 'CRR_{}(filter_sz={}, stride={}, do_res={})'.format(i,
                    filt_sz, stride, do_res)
            mod = ConvReLURes(in_chan, n_out, filt_sz, stride, do_res,
                    parent_rf, name)
            self.net.add_module(str(i), mod)
            parent_rf = mod.rf

        self.beg_rf = self.net[0].rf
        self.rf = parent_rf 

    def forward(self, mels):
        '''
        B, M, C, T = n_batch, n_mels, n_channels, n_timesteps 
        mels: (B, M, T) (torch.tensor)
        outputs: (B, C, T)
        '''
        assert self.beg_rf.src.nv == mels.shape[2]
        out = self.net(mels)
        assert self.rf.dst.nv == out.shape[2]
        return out

