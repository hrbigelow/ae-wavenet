# Miscellaneous functions for the network
import torch
from torch import nn
import vconv 
from sys import stderr
import sys
import re

def xavier_init(mod):
    if hasattr(mod, 'weight') and mod.weight is not None:
        nn.init.xavier_uniform_(mod.weight)
    if hasattr(mod, 'bias') and mod.bias is not None:
        nn.init.constant_(mod.bias, 0)


class EmbedLossAdjust(nn.Module):
    """
    Adjusts the set of embedding vectors by the degree of their usage
    in the decoder.  This adjustment is only necessary for the L2 loss
    terms, not the reconstruction loss.  The reason is that the reconstruction
    loss automatically propagates through the embedding vectors to the degree
    appropriate, due to the structure of the decoder.

    However, the usage of the same embedding vectors in the L2 and commitment
    loss terms does not flow through the decoder, so the degree of their usage
    is not accounted for.
    """
    def __init__(self, name=None):
        super(EmbedLossAdjust, self).__init__()
        self.name = name

    def init_usage_weight(self, beg_vc, end_vc, n_out):
        """
        Initialize the tensor of usage weights for the z terms for a window batch
        of timesteps.
        """
        total_in_b, total_in_e, __ = vconv.recep_field(beg_vc, end_vc, 0, n_out, n_out)
        assert total_in_b == 0
        self.register_buffer('z_usage', torch.zeros(total_in_e)) 
        for o in range(n_out):
            in_b, in_e, __ = vconv.recep_field(beg_vc, end_vc, o, o+1, n_out)
            self.z_usage[in_b:in_e] += 1
        self.z_usage /= torch.sum(self.z_usage)

    def forward(self, z_loss):
        """
        Carries out the adjustment
        """
        return self.z_usage * z_loss


class LCCombine(nn.Module):
    """
    The bottleneck loss terms for VAE and VQVAE are based on "z", which, for
    a single sample, consists of ~14 or so individual timesteps.  This enables
    the proper accounting of commitment loss terms when batching training
    samples across the time dimension.  
    NOTE: This module is not managed by save/restore, nor by initialize_weights.
    """
    def __init__(self, name=None):
        super(LCCombine, self).__init__()
        self.name = name

#    def set_geometry(self, beg_rf, end_rf):
#        """
#        Constructs the transpose convolution which mimics the usage pattern
#        of WaveNet's local conditioning vectors and output.  
#        """
#        self.rf = vconv.condensed(beg_rf, end_rf, self.name) 
#        self.rf.gen_stats(self.rf)
#        self.rf.init_nv(1)
#        stride = self.rf.stride_ratio.denominator
#        l_off, r_off = rfield.offsets(self.rf, self.rf)
#        filter_sz = l_off - r_off + 1
#        # pad_add = kernel_size - 1 - pad_arg (see torch.nn.ConvTranspose1d)
#        # => pad_arg = kernel_size - 1 - pad_add 
#        pad_add = max(self.rf.l_pad, self.rf.r_pad)
#        self.l_trim = pad_add - self.rf.l_pad
#        self.r_trim = pad_add - self.rf.r_pad
#        pad_arg = filter_sz - 1 - pad_add
#        self.tconv = nn.ConvTranspose1d(1, 1, filter_sz, stride, pad_arg, bias=False)
#        self.tconv.weight.requires_grad = False
#        nn.init.constant_(self.tconv.weight, 1.0 / self.rf.src.nv)

    def forward(self, z_metric):
        """
        Adjusts the loss terms for each z by the fraction each z is used in the
        decoder.  Ordinarily, such an adjustment would be unnecessary, if all
        gradients were flowing backwards from one loss term.  However, in this
        model, there are two additional loss terms that don't flow from the
        very end of the autoencoder.
        
        B, T, S: batch_sz, timesteps, less-frequent timesteps
        D: number of 
        z_metric: B, S, 1 
        output: B, T, 1
        """
        out_trim = out[:,self.l_trim:-self.r_trim or None,:]
        return out


this = sys.modules[__name__]
this.print_iter = 0
def set_print_iter(pos):
    this.print_iter = pos


def print_metrics(metrics, hdr_frequency):
    nlstrip = re.compile('\\n\s+')
    sep = ''
    h = ''
    s = ''
    d = dict(metrics)

    for k, v in d.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            v = v.item()
        if isinstance(v, int):
            fmt = '{:d}'
        elif isinstance(v, float):
            fmt = '{:.3f}'
        else:
            fmt = '{}' 
        val = nlstrip.sub(' ', fmt.format(v))
        s += sep + val
        h += sep + '{}'.format(k)
        sep = '\t'

    if this.print_iter % hdr_frequency == 0:
        print(h, file=stderr)

    print(s, file=stderr)
    this.print_iter += 1
    stderr.flush()

