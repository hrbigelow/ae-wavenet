import data
import model
import torch

class State(object):
    '''Encapsulates full state of training'''
    def __init__(self, step=0, pre_params=None, enc_params=None,
            bn_params=None, dec_params=None, sample_catalog=None,
            sample_rate=None, frac_perm_use=None):
        self.model = None
        self.data = None
        self.step = step
        self.pre_params = pre_params
        self.enc_params = enc_params
        self.bn_params = bn_params
        self.dec_params = dec_params
        self.sample_catalog = sample_catalog
        self.sample_rate = sample_rate
        self.frac_perm_use = frac_perm_use

    def build(self):
        if (self.sample_catalog is None
                or self.sample_rate is None
                or self.frac_perm_use is None):
            raise RuntimeError('Cannot build without fully initialized State')
        self.data = data.WavSlices(self.sample_catalog, self.sample_rate,
                self.frac_perm_use)

        if (self.pre_params is None
                or self.enc_params is None
                or self.bn_params is None
                or self.dec_params is None):
            raise RuntimeError('Cannot build without fully initialized State')
        self.dec_params['n_speakers'] = self.data.num_speakers()
        self.model = model.AutoEncoder(self.pre_params, self.enc_params, self.bn_params,
                self.dec_params)

    def load(self, ckpt_file):
        sinfo = torch.load(ckpt_file)
        self.step = sinfo['step']
        mstate = sinfo['model']
        self.pre_params = mstate['pre']
        self.enc_params = mstate['enc']
        self.bn_params = mstate['bn']
        self.dec_params = mstate['dec']

        dstate = sinfo['data']
        self.sample_catalog = dstate['sample_catalog']
        self.sample_rate = dstate['sample_rate']
        self.frac_perm_use = dstate['frac_perm_use']

        self.build()

        self.model.load_state_dict(mstate['state_dict'])
        self.data.load_state_dict(dstate['state_dict'])

    def save(self, ckpt_file):
        mstate = { 
                'pre': self.pre_params,
                'enc': self.enc_params,
                'bn': self.bn_params,
                'dec': self.dec_params,
                'state_dict': self.model.state_dict() 
                }
        dstate = {
                'sample_catalog': self.sample_catalog,
                'sample_rate': self.sample_rate,
                'frac_perm_use': self.frac_perm_use,
                'state_dict': self.data.state_dict()
                }
        state = { 'step': self.step, 'model': mstate, 'data': dstate }
        torch.save(state, ckpt_file)

        



