# Full Autoencoder model
from hashlib import md5
import numpy as np
from pickle import dumps
import torch
from torch import nn
from torch.nn.modules import loss
from scipy.cluster.vq import kmeans

import ae_bn
import mfcc
import vconv
import util
import vq_bn
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

        # A dummy buffer that simply allows querying the current model device 
        self.register_buffer('dummy_buf', torch.empty(0))

    def one_hot(self, wav_compand):
        """
        wav_compand: (B, T)
        B, Q, T: n_batch, n_quant, n_timesteps
        returns: (B, Q, T)
        """
        wav_one_hot = util.gather_md(self.quant_onehot, 0, wav_compand.long()).permute(1,0,2)
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
            self.bottleneck = vq_bn.VQEMA(**bn_extra, n_in=enc_params['n_out'],
                    training=training)
            self.objective = vq_bn.VQEMALoss(self.bottleneck)

        elif bn_type == 'vae':
            # mu and sigma members  
            self.bottleneck = vae_bn.VAE(**bn_extra, n_in=enc_params['n_out'])
            self.objective = vae_bn.SGVBLoss(self.bottleneck)

        elif bn_type == 'ae':
            self.bottleneck = ae_bn.AE(**bn_extra, n_in=enc_params['n_out'])
            self.objective = torch.nn.CrossEntropyLoss()

        else:
            raise InvalidArgument('bn_type must be one of "ae", "vae", or "vqvae"')

        self.bn_type = bn_type
        self.decoder = dec.WaveNet(
                **dec_params,
                parent_vc=self.encoder.vc,
                n_lc_in=bn_params['n_out']
                )
        self.vc = self.decoder.vc

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
        """Initialize the VQ Embedding with samples from the encoder."""
        if self.bn_type not in ('vqvae', 'vqvae-ema'):
            raise RuntimeError('init_vq_embed only applies to the vqvae model type')

        bn = self.bottleneck
        e = 0
        n_codes = bn.emb.shape[0]
        k = bn.emb.shape[1]
        samples = np.empty((n_samples, k), dtype=np.float) 
        
        with torch.no_grad():
            while e != n_samples:
                vbatch = data_source.next_batch()
                encoding = self.encoder(vbatch.mel_input)
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
        

    def forward(self, mels, wav_onehot_dec, voice_inds, lcond_slice):
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
        quant = self.decoder(wav_onehot_dec, encoding_bn, voice_inds,
                lcond_slice)
        return quant

    def run(self, vbatch):
        """Run the model on one batch, returning the predicted and
        actual output
        B, T, Q: n_batch, n_timesteps, n_quant
        Outputs:
        quant_pred: (B, Q, T) (the prediction from the model)
        wav_batch_out: (B, T) (the actual data from the same timesteps)
        """
        wav_onehot_dec = self.preprocess(vbatch.wav_input)

        # Slice each wav input
        wav_batch_out = vbatch.wav_input.new_empty(vbatch.batch_size,
                vbatch.loss_wav_len()) 
        for b, (sl_b, sl_e) in enumerate(vbatch.loss_wav_slice):
            wav_batch_out[b] = vbatch.wav_input[b,sl_b:sl_e]

        quant = self.forward(vbatch.mel_input, wav_onehot_dec,
                vbatch.voice_index, vbatch.lcond_slice)
        # quant_pred[:,:,0] is a prediction for wav_compand_out[:,1] 
        return quant[...,:-1], wav_batch_out[...,1:]

    def geometry(self):
        """

        """

class Metrics(object):
    """Manage running the model and saving output and target state"""
    def __init__(self, state):
        self.state = state
        self.quant = None
        self.target = None
        self.softmax = torch.nn.Softmax(1) # input to this is (B, Q, N)

    def update(self):
        data_batch = self.state.data.next_batch()
        quant_pred_snip, wav_compand_out_snip = self.state.model.run(data_batch) 
        self.quant = quant_pred_snip
        self.target = wav_compand_out_snip
        self.probs = self.softmax(self.quant)

    def loss(self):
        """This is the closure needed for the optimizer"""
        if self.quant is None or self.target is None:
            raise RuntimeError('Must call update() first')
        self.state.optim.zero_grad()
        loss = self.state.model.objective(self.quant, self.target)
        loss.backward()
        return loss
    
    def peak_dist(self):
        """Average distance between the indices of the peaks in pred and
        target"""
        diffs = torch.argmax(self.quant, dim=1) - self.target.long()
        mean = torch.mean(torch.abs(diffs).float())
        return mean

    def avg_max(self):
        """Average max value for the predictions.  As the prediction becomes
        more peaked, this should go up"""
        max_val, max_ind = torch.max(self.probs, dim=1)
        mean = torch.mean(max_val)
        return mean
        
    def avg_prob_target(self):
        """Average probability given to target"""
        target_probs = torch.gather(self.probs, 1, self.target.long().unsqueeze(1)) 
        mean = torch.mean(target_probs)
        return mean

