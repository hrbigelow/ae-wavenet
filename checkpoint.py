import torch
import pickle

class State(object):
    '''Encapsulates full state of training'''
    def __init__(self, step=0, model=None, data=None):
        self.model = model 
        self.data = data 
        self.step = step

    def load(self, ckpt_file):
        sinfo = torch.load(ckpt_file)
        self.model = pickle.loads(sinfo['model'])
        self.data = pickle.loads(sinfo['data'])
        self.step = sinfo['step']

    def save(self, ckpt_file):
        mstate = pickle.dumps(self.model)
        dstate = pickle.dumps(self.data)
        state = { 'step': self.step, 'model': mstate, 'data': dstate }
        torch.save(state, ckpt_file)
