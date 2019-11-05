import pickle
import torch
from sys import stderr
import util

class State(object):
    '''Encapsulates full state of training'''
    def __init__(self, step=0, model=None, data=None, optim=None):
        self.model = model 
        self.data = data 
        self.optim = optim
        self.step = step
        self.torch_rng_state = torch.get_rng_state()
        self.torch_cuda_rng_states = torch.cuda.get_rng_state_all()

    def load(self, ckpt_file):
        sinfo = torch.load(ckpt_file)
        self.model = pickle.loads(sinfo['model'])
        self.data = pickle.loads(sinfo['data'])
        self.model.encoder.set_parent_vc(self.data.mfcc_vc)
        self.optim = torch.optim.Adam(self.model.parameters())
        self.optim.load_state_dict(sinfo['optim'])
        self.step = sinfo['step']
        self.torch_rng_state = sinfo['rand_state']
        self.torch_cuda_rng_states = sinfo['cuda_rand_states']

    def save(self, ckpt_file):
        mstate = pickle.dumps(self.model)
        dstate = pickle.dumps(self.data)
        ostate = self.optim.state_dict()
        state = {
                'step': self.step,
                'model': mstate,
                'data': dstate,
                'optim': ostate,
                'rand_state': torch.get_rng_state(),
                'cuda_rand_states': torch.cuda.get_rng_state_all()
                }
        torch.save(state, ckpt_file)

    def to(self, device):
        """Hack to move both model and optimizer to device"""
        self.model.to(device)
        self.data.to(device)
        ostate = self.optim.state_dict()
        self.optim = torch.optim.Adam(self.model.parameters())
        self.optim.load_state_dict(ostate)

    def optim_checksum(self):
        return util.digest(self.optim.state_dict())

    def init_torch_generator(self):
        """Hack to set the generator state"""
        torch.set_rng_state(self.torch_rng_state)
        #print('saved generator state: {}'.format(
        #    util.tensor_digest(self.torch_cuda_rng_states)))
        #torch.cuda.set_rng_state_all(self.torch_cuda_rng_states)
        torch.cuda.set_rng_state(self.torch_cuda_rng_states[0])
        ndiff = torch.cuda.get_rng_state().ne(self.torch_cuda_rng_states[0]).sum()
        if ndiff != 0:
            print(('Warning: restored and checkpointed '
            'GPU state differs in {} positions').format(ndiff), file=stderr)
            stderr.flush()

    def update_learning_rate(self, learning_rate):
        sd = self.optim.state_dict()
        self.optim = torch.optim.Adam(params=self.model.parameters(),
                lr=learning_rate)
        self.optim.load_state_dict(sd)

