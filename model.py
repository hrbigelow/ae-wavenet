# Full Autoencoder model
from sys import stderr
from hashlib import md5
import numpy as np
from pickle import dumps
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

# from numpy import vectorize as np_vectorize
class PreProcess(nn.Module):
    """
    Perform one-hot encoding
    """
    def __init__(self, pre_params, n_quant):
        super(PreProcess, self).__init__()
        self.n_quant = n_quant
        self.register_buffer('quant_onehot', torch.eye(self.n_quant))

    def one_hot(self, wav_compand):
        """
        wav_compand: (B, T)
        B, Q, T: n_batch, n_quant, n_timesteps
        returns: (B, Q, T)
        """
        wav_compand_tmp = wav_compand.long()
        wav_one_hot = util.gather_md(self.quant_onehot, 0, wav_compand_tmp).permute(1,0,2)
        return wav_one_hot

    def forward(self, in_snd_slice):
        """
        Converts the input to a one-hot format
        """
        in_snd_slice_onehot = self.one_hot(in_snd_slice)
        return in_snd_slice_onehot


class AutoEncoder(nn.Module):
    """
    Full Autoencoder model.  The _initialize method allows us to seamlessly initialize
    from __init__ or __setstate__ 
    """
    def __init__(self, pre_params, enc_params, bn_params, dec_params,
            n_mel_chan, training):
        self.init_args = {
                'pre_params': pre_params,
                'enc_params': enc_params,
                'bn_params': bn_params,
                'dec_params': dec_params,
                'n_mel_chan': n_mel_chan,
                'training': training
                }
        self._initialize()

    def _initialize(self):
        super(AutoEncoder, self).__init__() 
        pre_params = self.init_args['pre_params']
        enc_params = self.init_args['enc_params']
        bn_params = self.init_args['bn_params']
        dec_params = self.init_args['dec_params']
        n_mel_chan = self.init_args['n_mel_chan']
        training = self.init_args['training']

        # the "preprocessing"
        self.preprocess = PreProcess(pre_params, n_quant=dec_params['n_quant'])

        self.encoder = enc.Encoder(n_in=n_mel_chan, parent_vc=None, **enc_params)

        bn_type = bn_params['type']
        bn_extra = dict((k, v) for k, v in bn_params.items() if k != 'type')
    
        # In each case, the objective function's 'forward' method takes the
        # same arguments.
        if bn_type == 'vqvae':
            self.bottleneck = vq_bn.VQ(**bn_extra, n_in=enc_params['n_out'])
            self.objective = vq_bn.VQLoss(self.bottleneck)

        elif bn_type == 'vqvae-ema':
            self.bottleneck = vqema_bn.VQEMA(**bn_extra, n_in=enc_params['n_out'],
                    training=training)
            self.objective = vqema_bn.VQEMALoss(self.bottleneck)

        elif bn_type == 'vae':
            # mu and sigma members  
            self.bottleneck = vae_bn.VAE(**bn_extra, n_in=enc_params['n_out'])
            self.objective = vae_bn.SGVBLoss(self.bottleneck)

        elif bn_type == 'ae':
            self.bottleneck = ae_bn.AE(n_out=bn_extra['n_out'], n_in=enc_params['n_out'])
            self.objective = ae_bn.AELoss(self.bottleneck, 0.001) 

        else:
            raise InvalidArgument('bn_type must be one of "ae", "vae", or "vqvae"')

        self.bn_type = bn_type
        self.decoder = dec.WaveNet(
                **dec_params,
                parent_vc=self.encoder.vc['end'],
                n_lc_in=bn_params['n_out']
                )
        self.vc = self.decoder.vc
        self.decoder.post_init()


    def __getstate__(self):
        state = { 
                'init_args': self.init_args,
                'state_dict': self.state_dict()
                }
        return state 

    def __setstate__(self, state):
        self.init_args = state['init_args']
        self._initialize()
        self.load_state_dict(state['state_dict'])


    def init_codebook(self, data_source, n_samples):
        """
        Initialize the VQ Embedding with samples from the encoder
        """
        if self.bn_type not in ('vqvae', 'vqvae-ema'):
            raise RuntimeError('init_vq_embed only applies to the vqvae model type')

        bn = self.bottleneck
        e = 0
        n_codes = bn.emb.shape[0]
        k = bn.emb.shape[1]
        samples = np.empty((n_samples, k), dtype=np.float) 
        
        with torch.no_grad():
            while e != n_samples:
                vbatch = next(data_source)
                encoding = self.encoder(vbatch.mel_enc_input)
                ze = self.bottleneck.linear(encoding)
                ze = ze.permute(0, 2, 1).flatten(0, 1)
                c = min(n_samples - e, ze.shape[0])
                samples[e:e + c,:] = ze.cpu()[0:c,:]
                e += c

        km, __ = kmeans(samples, n_codes)
        bn.emb[...] = torch.from_numpy(km)

        if self.bn_type == 'vqvae-ema':
            bn.ema_numer = bn.emb * bn.ema_gamma_comp
            bn.ema_denom = bn.n_sum_ones * bn.ema_gamma_comp
        
    def checksum(self):
        """Return checksum of entire set of model parameters"""
        return util.tensor_digest(self.parameters())
        

    def forward(self, mels, wav_onehot_dec, voice_inds, jitter_index,
            trim_ups_out):
        """
        B: n_batch
        M: n_mels
        T: receptive field of autoencoder
        T': receptive field of decoder 
        R: size of local conditioning output of encoder (T - encoder.vc.total())
        N: n_win (# consecutive samples processed in one batch channel)
        Q: n_quant
        mels: (B, M, T)
        wav_compand: (B, T)
        wav_onehot_dec: (B, T')  
        Outputs: 
        quant_pred (B, Q, N) # predicted wav amplitudes
        """
        encoding = self.encoder(mels)
        encoding_bn = self.bottleneck(encoding)
        self.encoding_bn = encoding_bn
        quant = self.decoder(wav_onehot_dec, encoding_bn, voice_inds,
                jitter_index, trim_ups_out)
        return quant

    def run(self, vbatch):
        """
        Run the model on one batch, returning the predicted and
        actual output
        B, T, Q: n_batch, n_timesteps, n_quant
        Outputs:
        quant_pred: (B, Q, T) (the prediction from the model)
        wav_batch_out: (B, T) (the actual data from the same timesteps)
        """
        wav_onehot_dec = self.preprocess(vbatch.wav_dec_input)
        # grad = torch.autograd.grad(wav_onehot_dec, vbatch.wav_dec_input).data

        # Slice each wav input
        trim = vbatch.ds.trim_dec_out
        wav_batch_out = vbatch.wav_dec_input[:,trim[0]:trim[1]]
        # wav_batch_out = torch.take(vbatch.wav_dec_input, vbatch.loss_wav_slice)
        #for b, (sl_b, sl_e) in enumerate(vbatch.loss_wav_slice):
        #    wav_batch_out[b] = vbatch.wav_dec_input[b,sl_b:sl_e]

        # self.wav_batch_out = wav_batch_out
        self.wav_onehot_dec = wav_onehot_dec

        quant = self.forward(vbatch.mel_enc_input, wav_onehot_dec,
                vbatch.voice_index, vbatch.jitter_index, vbatch.ds.trim_ups_out)
        # quant_pred[:,:,0] is a prediction for wav_compand_out[:,1] 
        return quant[...,:-1], wav_batch_out[...,1:]


class GPULoaderIter(object):
    def __init__(self, data_iter):
        self.data_iter = data_iter

    def __next__(self):
        return self.data_iter.__next__()[0]


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

    def __init__(self, mode, opts):
        print('Initializing model and data source...', end='', file=stderr)
        stderr.flush()
        self.opts = opts

        torch.manual_seed(opts.random_seed)
        pre_par = parse_tools.get_prefixed_items(vars(opts), 'pre_')
        enc_par = parse_tools.get_prefixed_items(vars(opts), 'enc_')
        bn_par = parse_tools.get_prefixed_items(vars(opts), 'bn_')
        dec_par = parse_tools.get_prefixed_items(vars(opts), 'dec_')

        # Initialize data
        jprob = dec_par.pop('jitter_prob')
        dataset = data.Slice(opts.n_batch, opts.n_win_batch, jprob,
                pre_par['sample_rate'], pre_par['mfcc_win_sz'],
                pre_par['mfcc_hop_sz'], pre_par['n_mels'],
                pre_par['n_mfcc'])
        dataset.load_data(opts.dat_file)
        # dec_par['n_speakers'] = dataset.num_speakers()
        # model = ae.AutoEncoder(pre_par, enc_par, bn_par, dec_par,
        #         dataset.num_mel_chan(), training=True)
        # model.encoder.set_parent_vc(dataset.mfcc_vc)
        # dataset.post_init(model.encoder.vc, model.decoder.vc)

        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        self.device = xm.xla_device()
        self.data_loader = pl.ParallelLoader(self.state.data_loader, [self.device])
        self.data_iter = TPULoaderIter(self.data_loader, self.device)

        self.state.init_torch_generator()
        print('Done.', file=stderr)
        stderr.flush()


    def train(self, index):
        batch_pre = next(self.data_iter)
        batch = next(self.data_iter)

