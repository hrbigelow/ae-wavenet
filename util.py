import numpy as np
import torch

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
    mu = torch.tensor(float(n_quanta - 1), device=quanta.device)
    qf = quant.to(dtype=torch.float32)
    inv_mu = mu.reciprocal()
    a = (2 * qf - 1) * inv_mu - 1
    x = torch.sign(a) * ((1 + mu)**torch.abs(a) - 1) * inv_mu
    return x

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
    if not 0 <= dim < len(input.size()):
        raise ValueError('dim {} must be in [0, {})'.format(dim, len(input.size())))

    # Q = prod(q_(1..m))
    # x: (i_1..i_k Q/i_d)
    k = len(input.size())
    x = torch.index_select(input, dim, query.flatten())

    # x_perm: (i_1..i_k / q) + Q.  In other words, move dimension Q to the end
    x_perm = x.permute(tuple(range(dim)) + tuple(range(dim+1, k)) + (dim,))

    # for example, expand (i_1, i_2, i_3, Q) to (i_1, i_2, i_3, q_1, q_2, q_3)
    out_size = input.size()[:dim] + input.size()[dim+1:] + query.size()
    return x_perm.reshape(out_size) 

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

