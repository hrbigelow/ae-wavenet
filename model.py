# Full Autoencoder model
from sys import stderr
from hashlib import md5
import numpy as np
from pickle import dumps
import pickle
import torch
from torch import nn
from torch.nn.modules import loss
from scipy.cluster.vq import kmeans

import model as ae
import checkpoint
import ae_bn
import data
import mfcc
import parse_tools  
import vconv
import util
import netmisc
import vq_bn
import vqema_bn
import vae_bn
import wave_encoder as enc
import wavenet as dec 

class TPULoaderIter(object):
    def __init__(self, parallel_loader, device):
        self.per_dev_loader = parallel_loader.per_device_loader(device)

    def __next__(self):
        vb = self.per_dev_loader.__next__()[0]
        return vb


class Metrics(object):
    """
    Manage running the model and saving output and target state
    """

    def __init__(self, opts):

        # Initialize data
        dataset = data.Slice(10, 1000)
        
        dataset.extra_field = torch.ByteTensor(np.random.rand(11338))

        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        self.device = xm.xla_device()
        wav_loader = data.WavLoader(dataset)
        self.data_loader = pl.ParallelLoader(wav_loader, [self.device])
        self.data_iter = TPULoaderIter(self.data_loader, self.device)


    def train(self, index):
        batch_pre = next(self.data_iter)
        batch = next(self.data_iter)


