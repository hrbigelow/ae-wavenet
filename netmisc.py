# Miscellaneous functions for the network
from torch import nn
import rfield

def _xavier_init(mod):
    nn.init.xavier_uniform_(mod.weight)
    if mod.bias is not None:
        nn.init.constant_(mod.bias, 0)


class LCCombine(nn.Module):
    '''The bottleneck loss terms for VAE and VQVAE are based on "z", which, for
    a single sample, consists of ~14 or so individual timesteps.  This enables
    the proper accounting of commitment loss terms when batching training
    samples across the time dimension.  
    NOTE: This module is not managed by save/restore, nor by initialize_weights.
    '''
    def __init__(self, name=None):
        super(LCCombine, self).__init__()
        self.name = name

    def set_geometry(self, beg_rf, end_rf):
        '''
        Constructs the transpose convolution which mimics the usage pattern
        of WaveNet's local conditioning vectors and output.
        '''
        self.rf = rfield.condensed(beg_rf, end_rf, self.name) 
        self.rf.gen_stats(self.rf)
        stride = self.rf.stride_ratio.denominator
        l_off, r_off = rfield.offsets(self.rf, self.rf)
        filter_sz = l_off - r_off + 1
        # pad_add = kernel_size - 1 - pad_arg (see torch.nn.ConvTranspose1d)
        # => pad_arg = kernel_size - 1 - pad_add 
        pad_add = max(self.rf.l_pad, self.rf.r_pad)
        self.l_trim = pad_add - self.rf.l_pad
        self.r_trim = pad_add - self.rf.r_pad
        pad_arg = filter_sz - 1 - pad_add
        self.tconv = nn.ConvTranspose1d(1, 1, filter_sz, stride, pad_arg, bias=False)
        self.tconv.weight.requires_grad = False
        nn.init.constant_(self.tconv.weight, 1.0)

    def forward(self, z_metric):
        '''
        B, T, S: batch_sz, timesteps, less-frequent timesteps
        D: number of 
        z_metric: B, S, 1 
        output: B, T, 1
        '''
        out = self.tconv(z_metric)
        out_trim = out[:,self.l_trim:-self.r_trim or None,:]
        return out_trim

