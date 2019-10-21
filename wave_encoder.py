import torch
from torch import nn
import vconv
import numpy as np
import netmisc

class ConvReLURes(nn.Module):
    def __init__(self, n_in_chan, n_out_chan, filter_sz, stride=1, do_res=True,
            parent_vc=None, name=None):
        super(ConvReLURes, self).__init__()
        self.do_res = do_res
        if self.do_res:
            if stride != 1:
                print('Stride must be 1 for residually connected convolution',
                        file=sys.stderr)
                raise ValueError

        self.n_in = n_in_chan
        self.n_out = n_out_chan
        self.conv = nn.Conv1d(n_in_chan, n_out_chan, filter_sz, stride, padding=0, bias=False)
        self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm1d(n_out_chan)

        self.vc = vconv.VirtualConv(filter_info=filter_sz, stride=stride,
                parent=parent_vc, name=name)
        netmisc.xavier_init(self.conv)

    def forward(self, x):
        '''
        B, C, T = n_batch, n_in_chan, n_win
        x: (B, C, T)
        '''
        out = self.conv(x)
        # out = self.bn(out)
        out = self.relu(out)
        if (self.do_res):
            win_size = x.shape[2] 
            shadow_b, shadow_e = vconv.shadow(self.vc, self.vc, 0, win_size, win_size) 
            e_off = win_size - shadow_e
            out += x[:,:,shadow_b:e_off or None]
        return out


class Encoder(nn.Module):
    def __init__(self, n_in, n_out, parent_vc):
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
                    parent_vc, name)
            self.net.add_module(str(i), mod)
            parent_vc = mod.vc

        self.beg_vc = self.net[0].vc
        self.vc = parent_vc 

    def forward(self, mels):
        '''
        B, M, C, T = n_batch, n_mels, n_channels, n_timesteps 
        mels: (B, M, T) (torch.tensor)
        outputs: (B, C, T)
        '''
        out = self.net(mels)
        #out = torch.tanh(out * 10.0)
        return out

