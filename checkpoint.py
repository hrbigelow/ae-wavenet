import pickle
import io
import torch
from sys import stderr
import util
import model as ae
import data

class State(object):
    '''Encapsulates full state of training'''
    def __init__(self, step=0, model=None, dataset=None, optim=None):
        self.model = model 
        self.data_loader = data.WavLoader(dataset)
        self.optim = optim
        self.step = step
        self.device = None
        self.torch_rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            self.torch_cuda_rng_states = torch.cuda.get_rng_state_all()
        else:
            self.torch_cuda_rng_states = None


    def load(self, ckpt_file, dat_file):
        sinfo = torch.load(ckpt_file)

        # This is the required order for model and data init 
        self.model = pickle.loads(sinfo['model'])
        dataset = pickle.loads(sinfo['dataset'])
        dataset.load_data(dat_file)
        self.model.post_init(dataset)
        self.model.load_state_dict(sinfo['model_state_dict'])
        dataset.post_init(self.model)

        self.data_loader = data.WavLoader(dataset)
        self.optim = torch.optim.Adam(self.model.parameters())
        self.optim.load_state_dict(sinfo['optim'])
        self.step = sinfo['step']
        self.torch_rng_state = sinfo['rand_state']
        self.torch_cuda_rng_states = sinfo['cuda_rand_states']

    def save(self, ckpt_file):
        # cur_device = self.device
        # self.to(torch.device('cpu'))

        mstate = pickle.dumps(self.model)
        mstate_dict = self.model.state_dict()
        dstate = pickle.dumps(self.data_loader.dataset)
        ostate = self.optim.state_dict()
        state = {
                'step': self.step,
                'model': mstate,
                'model_state_dict': mstate_dict,
                'dataset': dstate,
                'optim': ostate,
                'rand_state': torch.get_rng_state(),
                'cuda_rand_states': (torch.cuda.get_rng_state_all() if
                    torch.cuda.is_available() else None)
                }
        if self.device.type == 'cuda':
            torch.save(state, ckpt_file)
        elif self.device.type == 'xla':
            import torch_xla.core.xla_model as xm
            xm.save(state, ckpt_file)
        # self.to(cur_device)

    def to(self, device):
        """Hack to move both model and optimizer to device"""
        self.device = device
        self.model.to(device)
        ostate = self.optim.state_dict()
        self.optim = torch.optim.Adam(self.model.parameters())
        self.optim.load_state_dict(ostate)
        self.device = device

    def optim_checksum(self):
        return util.digest(self.optim.state_dict())

    def init_torch_generator(self):
        """Hack to set the generator state"""
        torch.set_rng_state(self.torch_rng_state)
        #print('saved generator state: {}'.format(
        #    util.tensor_digest(self.torch_cuda_rng_states)))
        #torch.cuda.set_rng_state_all(self.torch_cuda_rng_states)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(self.torch_cuda_rng_states[0])
            ndiff = torch.cuda.get_rng_state().ne(self.torch_cuda_rng_states[0]).sum()
            if ndiff != 0:
                print(('Warning: restored and checkpointed '
                'GPU state differs in {} positions').format(ndiff), file=stderr)
                stderr.flush()

    def update_learning_rate(self, learning_rate):
        for g in self.optim.param_groups:
            g['lr'] = learning_rate


