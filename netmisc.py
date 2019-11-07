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


this = sys.modules[__name__]
this.print_iter = 0
def set_print_iter(pos):
    this.print_iter = pos


def print_metrics(metrics, hdr_frequency):
    """
    Flexibly prints a polymorphic set of metrics
    """
    nlstrip = re.compile('\\n\s+')
    sep = ''
    h = ''
    s = ''
    d = dict(metrics)
    max_width = 7

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
        if len(val) > max_width:
            val = '~' + val[-(max_width-1):]
            
        s += sep + val
        h += sep + '{}'.format(k)
        sep = '\t'

    if this.print_iter % hdr_frequency == 0:
        print(h, file=stderr)

    print(s, file=stderr)
    this.print_iter += 1
    stderr.flush()

