from hashlib import md5
from pickle import dumps
import numpy as np
import torch
from typing import Tuple

def digest(obj):
    return md5(dumps(obj)).hexdigest()

def tensor_digest(tensors):
    try:
        it = iter(tensors)
    except TypeError:
        tensors = list(tensors)

    vals = list(map(lambda t: t.flatten().detach().cpu().numpy().tolist(), tensors))
    return digest(vals)

def _validate_checkpoint_info(ckpt_dir, ckpt_file_template):
    # Unfortunately, Python doesn't provide a way to hold an open directory
    # handle, so we just check whether the directory path exists and is
    # writable during this call.
    import os
    if not os.access(ckpt_dir, os.R_OK|os.W_OK):
        raise ValueError('Cannot read and write checkpoint directory {}'.format(ckpt_dir))
    # test if ckpt_file_template is valid  
    try:
        test_file = ckpt_file_template.replace('%', '1000')
    except IndexError:
        test_file = ''
    # '1000' is 3 longer than '%'
    if len(test_file) != len(ckpt_file_template) + 3:
        raise ValueError('Checkpoint template "{}" ill-formed. ' 
                '(should have exactly one "%")'.format(ckpt_file_template))
    try:
        test_path = '{}/{}'.format(ckpt_dir, test_file)
        if not os.access(test_path, os.R_OK):
            fp = open(test_path, 'w')
            fp.close()
            os.remove(fp.name)
    except IOError:
        raise ValueError('Cannot create a test checkpoint file {}'.format(test_path))


class CheckpointPath(object):
    def __init__(self, path_template):
        import os.path
        _dir = os.path.dirname(path_template) 
        _base = os.path.basename(path_template)
        if _dir == '' or _base == '':
            raise ValueError('path_template "{}" does not contain both '
                    'directory and file'.format(path_template))
        self.dir = _dir.rstrip('/')
        self.file_template = _base 
        _validate_checkpoint_info(self.dir, self.file_template)

    def path(self, step):
        return '{}/{}'.format(self.dir, self.file_template.replace('%', str(step)))


def mu_encode_np(x, n_quanta):
    '''mu-law encode and quantize'''
    mu = n_quanta - 1
    amp = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    quant = (amp + 1) * 0.5 * mu + 0.5
    return quant.astype(np.int32)


def mu_decode_np(quant, n_quanta):
    '''accept an integer mu-law encoded quant, and convert
    it back to the pre-encoded value'''
    mu = n_quanta - 1
    qf = quant.astype(np.float32)
    inv_mu = 1.0 / mu
    a = (2 * qf - 1) * inv_mu - 1
    x = np.sign(a) * ((1 + mu)**np.fabs(a) - 1) * inv_mu
    return x


def mu_encode_torch(x, n_quanta):
    '''mu-law encode and quantize'''
    mu = torch.tensor(float(n_quanta - 1), device=x.device)
    amp = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
    quant = (amp + 1) * 0.5 * mu + 0.5
    return quant.round_().to(dtype=torch.long)

def mu_decode_torch(quant, n_quanta):
    '''accept an integer mu-law encoded quant, and convert
    it back to the pre-encoded value'''
    mu = torch.tensor(float(n_quanta - 1), device=quant.device)
    qf = quant.to(dtype=torch.float32)
    inv_mu = mu.reciprocal()
    a = (2 * qf - 1) * inv_mu - 1
    x = torch.sign(a) * ((1 + mu)**torch.abs(a) - 1) * inv_mu
    return x

def entropy(ten, do_norm=True):
    if do_norm:
        s = ten.sum()
        n = ten / s
    else:
        n = ten
    lv = torch.where(n == 0, n.new_zeros(n.size()), torch.log2(n))
    return - (n * lv).sum()

def int_hist(ten, ignore_val=None, accu=None):
    """Return a histogram of the integral-valued tensor"""
    if ten.is_floating_point():
        raise RuntimeError('int_hist only works for non-floating-point tensors')

    if ignore_val is not None:
        mask = ten.ne(ignore_val)
        ten = ten.masked_select(mask)

    ne = max(ten.max() + 1, ten.nelement())
    o = ten.new_ones(ne, dtype=torch.float)
    if accu is None:
        z = o.new_zeros(ne)
    else:
        z = accu
    z.scatter_add_(0, ten.flatten(), o) 
    return z


"""
torch.index_select(input, d, query), expressed as SQL:

    d: integer in (1..k)
    input: i_(1..k), ival
    query: q_1, qval

    SELECT (i_1..i_k q_1/i_d), ival
    from input, query
    where i_d = qval

    notation: (1..k q/d) means "values 1 through k, replacing d with q"
"""

"""
torch.gather(input, d, query), expressed as SQL:
    d: integer in (1..k)
    input: i_(1..k), ival
    query: q_(1..k), qval
    NOTE: max(q_j) = max(i_j) for all j != d

    SELECT (i_1 .. i_k qval/i_d), ival
    from index, query
    where i_d = qval

    The output has the same shape as query.
    All values of the output are values from input.
    It's like a multi-dimensional version of torch.take.
"""

# !!! this doesn't generalize to other dimensions anymore
def gather_md_jit(input, dim: int, perm: Tuple[int, int], query):
    """
    torchscript jit version
    """
    k = input.dim()
    if dim < 0 or dim >= k:
        raise ValueError('dim {} must be in [0, {})'.format(dim, k))

    # Q = prod(q_(1..m))
    # x: (i_1..i_k Q/i_d)
    x = torch.index_select(input, dim, query.flatten())

    # print('type of dim is: ', type(dim))
    # x_perm: (i_1..i_k / q) + Q.  In other words, move dimension Q to the end
    # t = list(range(dim)) + list(range(dim+1, k)) + [dim]
    # t = (0,1)
    # t = tuple(range(dim)) + tuple(range(dim+1, k)) + (dim,)
    # x_perm = x.permute(*t)
    # !!! original
    # t = tuple(range(dim)) + tuple(range(dim+1, k)) + (dim,)
    # print('permutation:', *perm)
    x_perm = x.permute(*perm)

    # for example, expand (i_1, i_2, i_3, Q) to (i_1, i_2, i_3, q_1, q_2, q_3)
    out_size = input.size()[:dim] + input.size()[dim+1:] + query.size()
    return x_perm.reshape(out_size) 


def gather_md(input, dim, query):
    '''
    You can view a K-dimensional tensor entry: input[i1,i2,...,ik] = cell_value
    as a SQL table record with fields        : i1, i2, ..., ik, cell_value
    
    Then, this function logically executes the following query:

    d: integer in (1..k)
    input: i_1, i_2, ..., i_k, ival
    query: q_1, q_2, ..., q_m, qval

    SELECT i_(1..k / d), q_(1..m), ival
    from input, query
    where i_d = qval

    (1..k / d) means "values 1 through k, excluding d"

    It is the same as torch.index_select, except that 'query' may have more
    than one dimension, and its dimension(s) are placed at the end of the
    result tensor rather than replacing input dimension 'dim'
    '''
    k = input.dim()
    tup = tuple(range(dim)) + tuple(range(dim+1, k)) + (dim,)
    return gather_md_scriptable(input, dim, tup, query)


def greatest_lower_bound(a, q): 
    '''return largest i such that a[i] <= q.  assume a is sorted.
    if q < a[0], return -1'''
    l, u = 0, len(a) - 1 
    while (l < u): 
        m = u - (u - l) // 2 
        if a[m] <= q: 
            l = m 
        else: 
            u = m - 1 
    return l or -1 + (a[l] <= q) 



def sigfig(f, s, m):
    """format a floating point value in fixed point notation but
    with a fixed number of significant figures.
    Examples with nsigfig=3, maxwidth=
    Rule is:
    1. If f < 1.0e-s, render with {:.ne} where n = s-1
    2. If f > 1.0e+l, render with {:.ne} where n = s-1
    3. Otherwise, render with {:0.ge} where g is:
       s if f in (0.1, 

         f         {:2e}       final
    1.23456e-04 => 1.23e-04 => unchanged
    1.23456e-03 => 1.23e-03 => unchanged
    1.23456e-02 => 1.23e-02 => unchanged 
    1.23456e-01 => 1.23e-01 => 0.123
    1.23456e+00 => 1.23e+00 => 1.230
    1.23456e+01 => 1.23e+01 => 12.30 
    1.23456e+02 => 1.23e+02 => 123.0
    1.23456e+03 => 1.23e+03 => 1230.
    1.23456e+04 => 1.23e+04 => 12300
    1.23456e+05 => 1.23e+05 => unchanged
    1.23456e+06 => 1.23e+06 => unchanged

    """
    pass

