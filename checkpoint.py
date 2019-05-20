import torch
import pickle
import util

class State(object):
    '''Encapsulates full state of training'''
    def __init__(self, step=0, model=None, data=None, optim=None):
        self.model = model 
        self.data = data 
        self.optim = optim
        self.step = step
        self.torch_rng_state = torch.get_rng_state()

    def load(self, ckpt_file):
        sinfo = torch.load(ckpt_file)
        self.model = pickle.loads(sinfo['model'])
        self.data = pickle.loads(sinfo['data'])
        self.optim = torch.optim.Adam(self.model.parameters())
        self.optim.load_state_dict(sinfo['optim'])
        self.step = sinfo['step']
        self.torch_rng_state = sinfo['rand_state']

    def save(self, ckpt_file):
        mstate = pickle.dumps(self.model)
        dstate = pickle.dumps(self.data)
        ostate = self.optim.state_dict()
        state = {
                'step': self.step,
                'model': mstate,
                'data': dstate,
                'optim': ostate,
                'rand_state': torch.get_rng_state()
                }
        torch.save(state, ckpt_file)

    def to(self, device):
        """Hack to move both model and optimizer to device"""
        self.model.to(device)
        ostate = self.optim.state_dict()
        self.optim = torch.optim.Adam(self.model.parameters())
        self.optim.load_state_dict(ostate)


    def optim_checksum(self):
        return util.digest(self.optim.state_dict())

    def init_torch_generator(self):
        """Hack to set the generator state"""
        torch.set_rng_state(self.torch_rng_state)

    def update_learning_rate(self, learning_rate):
        sd = self.optim.state_dict()
        self.optim = torch.optim.Adam(params=self.model.parameters(),
                lr=learning_rate)
        self.optim.load_state_dict(sd)
