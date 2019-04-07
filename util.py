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
        test_file = ckpt_file_template.format(1000)
    except IndexError:
        test_file = ''
    # '1000' is 2 longer than '{}'
    if len(test_file) != len(ckpt_file_template) + 2:
        raise ValueError('Checkpoint template "{}" ill-formed. ' 
                '(should have exactly one "{{}}")'.format(ckpt_file_template))
    try:
        test_path = '{}/{}'.format(ckpt_dir, test_file)
        if not os.access(test_path, os.R_OK):
            fp = open(test_path, 'w')
            fp.close()
            os.remove(fp.name)
    except IOError:
        raise ValueError('Cannot create a test checkpoint file {}'.format(test_path))


class CheckpointPath(object):
    def __init__(self):
        self.dir = None
        self.file_template = None
        self.enabled = False

    def enable(self, _dir, file_template):
        _validate_checkpoint_info(_dir, file_template)
        self.dir = _dir
        self.file_template = file_template
        self.enabled = True

    def path(self, step):
        if not self.enabled:
            raise RuntimeError('Must call enable first.')
        return '{}/{}'.format(self.dir, self.file_template.format(step))


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

def gather_md(input, dim, index):
    '''
    Creats a new tensor by replacing each scalar value s in index[...]
    with a subtensor input[:,:,...,s,...], where s is the dim'th dimension.

    The resulting gathered tensor has dimensions:

    **index.shape + **input_shape_without_dim
    '''
    x = torch.index_select(input, dim, index.flatten())
    input_shape = list(input.shape)
    index_shape = list(index.shape)
    output_shape = index_shape + [input_shape[i] for i in range(len(input_shape)) if i != dim]
    return x.reshape(output_shape) 

