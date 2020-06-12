import pickle
import io
import torch as t
from sys import stderr
import util
import data
import mfcc_inverter as mi
import hparams

class Checkpoint(object):
    '''
    Encapsulates full state of training
    '''

    def __init__(self, override_hps, dat_file, train_mode=True, ckpt_file=None,
            num_replicas=1, rank=0):
        """
        Initialize total state.  If ckpt_file is provided, hps will be ignored. 
        """
        if ckpt_file is not None:
            ckpt = t.load(ckpt_file)
            if 'hps' in ckpt:
                hps = hparams.Hyperparams(**ckpt['hps'])
        else:
            hps = hparams.Hyperparams()
        hps.update(override_hps)
            
        t.manual_seed(hps.random_seed)

        if hps.global_model == 'autoencoder':
            self.model = ae.AutoEncoder(hps)
        elif hps.global_model == 'mfcc_inverter':
            self.model = mi.MfccInverter(hps)

        slice_size = self.model.get_input_size(hps.n_win_batch)

        self.data = data.DataProcessor(hps, dat_file, self.model.mfcc,
                slice_size, train_mode, start_epoch=0, start_step=0,
                num_replicas=num_replicas, rank=rank)

        self.model.override(hps.n_win_batch)

        if ckpt_file is None:
            self.optim = t.optim.Adam(params=self.model.parameters(),
                    lr=hps.learning_rate_rates[0])

        else:
            sub_state = { k: v for k, v in ckpt['model_state_dict'].items() if
                    '_lead' not in k and 'left_wing_size' not in k }
            self.model.load_state_dict(sub_state, strict=False)
            if 'epoch' in ckpt:
                self.data.dataset.set_pos(ckpt['epoch'], ckpt['step'])
            else:
                global_step = ckpt['step']
                epoch = global_step // len(self.data.dataset)
                step = global_step % len(self.data.dataset) 
                self.data.dataset.set_pos(epoch, step)
                
            self.optim = t.optim.Adam(self.model.parameters())
            self.optim.load_state_dict(ckpt['optim'])
            self.torch_rng_state = ckpt['rand_state']
            self.torch_cuda_rng_states = ckpt['cuda_rand_states']

        
        self.device = None
        self.torch_rng_state = t.get_rng_state()
        if t.cuda.is_available():
            self.torch_cuda_rng_states = t.cuda.get_rng_state_all()
        else:
            self.torch_cuda_rng_states = None

        self.hps = hps


    def save(self, ckpt_file):
        # cur_device = self.device
        # self.to(t.device('cpu'))
        mstate_dict = self.model.state_dict()
        ostate = self.optim.state_dict()
        state = {
                'hps': self.hps,
                'epoch': self.data.epoch,
                'step': self.data.step,
                'model_state_dict': mstate_dict,
                'optim': ostate,
                'rand_state': t.get_rng_state(),
                'cuda_rand_states': (t.cuda.get_rng_state_all() if
                    t.cuda.is_available() else None)
                }
        if self.device.type == 'cuda':
            t.save(state, ckpt_file)
        elif self.device.type == 'xla':
            import torch_xla.core.xla_model as xm
            xm.save(state, ckpt_file)
        # self.to(cur_device)

    def to(self, device):
        """Hack to move both model and optimizer to device"""
        self.device = device
        self.model.to(device)
        ostate = self.optim.state_dict()
        self.optim = t.optim.Adam(self.model.parameters())
        self.optim.load_state_dict(ostate)
        self.device = device

    def optim_checksum(self):
        return util.digest(self.optim.state_dict())

    def init_torch_generator(self):
        """Hack to set the generator state"""
        t.set_rng_state(self.torch_rng_state)
        #print('saved generator state: {}'.format(
        #    util.tensor_digest(self.torch_cuda_rng_states)))
        #t.cuda.set_rng_state_all(self.torch_cuda_rng_states)
        if t.cuda.is_available():
            if self.torch_cuda_rng_states is not None:
                t.cuda.set_rng_state(self.torch_cuda_rng_states[0])
                ndiff = t.cuda.get_rng_state().ne(self.torch_cuda_rng_states[0]).sum()
                if ndiff != 0:
                    print(('Warning: restored and checkpointed '
                    'GPU state differs in {} positions').format(ndiff), file=stderr)
                    stderr.flush()

    def update_learning_rate(self, learning_rate):
        for g in self.optim.param_groups:
            g['lr'] = learning_rate


class InferenceState(object):
    """
    Restores a trained model for inference
    """

    def __init__(self, model=None, dataset=None):
        self.model = model 
        self.data_loader = data.WavLoader(dataset)
        self.device = None
        self.torch_rng_state = t.get_rng_state()
        if t.cuda.is_available():
            self.torch_cuda_rng_states = t.cuda.get_rng_state_all()
        else:
            self.torch_cuda_rng_states = None

    def to(self, device):
        self.device = device
        self.model.to(device)

    def load(self, ckpt_file, dat_file):
        ckpt = t.load(ckpt_file)

        # This is the required order for model and data init 
        self.model = pickle.loads(ckpt['model'])

        # win batch of 1 is inference mode
        self.model.override(n_win_batch=1)

        # ignore the pickled dataset characteristics
        dataset = data.MfccInference(pickle.loads(ckpt['dataset']), dat_file)

        # dataset.load_data(dat_file)
        self.model.post_init(dataset)
        sub_state = { k: v for k, v in ckpt['model_state_dict'].items() if '_lead' not
                in k and 'left_wing_size' not in k }
        self.model.load_state_dict(sub_state, strict=False)
        dataset.post_init(self.model)
        self.data_loader = data.WavLoader(dataset)

