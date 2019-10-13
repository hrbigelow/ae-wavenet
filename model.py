# Full Autoencoder model
from hashlib import md5
import numpy as np
from pickle import dumps
import torch
from torch import nn
from torch.nn.modules import loss

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
    '''Shape tensors by appropriate offsets to feed to Loss function'''
    def __init__(self, pre_params, n_quant):
        super(PreProcess, self).__init__()
        self.mfcc = mfcc.ProcessWav(**pre_params, name='mfcc')
        self.vc = self.mfcc.vc
        self.n_quant = n_quant
        self.register_buffer('quant_onehot', torch.eye(self.n_quant))

        # A dummy buffer that simply allows querying the current model device 
        self.register_buffer('dummy_buf', torch.empty(0))

    def set_geometry(self, enc_off, dec_off):
        '''
        '''
        self.l_enc_off, self.r_enc_off = enc_off
        self.l_dec_off, self.r_dec_off = dec_off 


    def one_hot(self, wav_compand):
        '''
        wav_compand: (B, T)
        B, Q, T: n_batch, n_quant, n_timesteps
        returns: (B, Q, T)
        '''
        wav_one_hot = util.gather_md(self.quant_onehot, 0, wav_compand.long()).permute(1,0,2)
        return wav_one_hot


    def forward(self, inds_np, wav_np):
        '''Inputs:
        B, M, Q: n_batch, n_mels, n_quant
        T: n_timesteps, receptive field of decoder 
        T': n_timesteps, output size of decoder
        inds_np: (B) (numpy)
        wav_np: (B, T)
        Outputs:
        inds: (B) (torch.tensor on current device)
        mels: (B,  
        wav_onehot_dec: (B, Q, T) (input to decoder)
        wav_compand_out: (B, T') (input matching the timestep range of decoder output)
        '''
        mels_np = np.apply_along_axis(self.mfcc.func, axis=1, arr=wav_np)

        # First moving of tensors to the destination device
        mels = torch.tensor(mels_np, device=self.dummy_buf.device)
        wav = torch.tensor(wav_np, device=self.dummy_buf.device)
        inds = torch.tensor(inds_np, device=self.dummy_buf.device)

        wav_dec = wav[:,self.l_enc_off:self.r_enc_off or None]
        wav_compand_dec = util.mu_encode_torch(wav_dec, self.n_quant)
        wav_compand_out = wav_compand_dec[:, self.l_dec_off:self.r_dec_off or None]
        wav_onehot_dec = self.one_hot(wav_compand_dec)
        return inds, mels, wav_onehot_dec, wav_compand_out


class AutoEncoder(nn.Module):
    '''
    Full Autoencoder model
    '''
    def __init__(self, pre_params, enc_params, bn_params, dec_params, sam_per_slice):
        self.args = [pre_params, enc_params, bn_params, dec_params, sam_per_slice]
        self._initialize()

    def _initialize(self):
        super(AutoEncoder, self).__init__() 
        pre_params, enc_params, bn_params, dec_params, sam_per_slice = self.args

        # the "preprocessing"
        self.preprocess = PreProcess(pre_params, n_quant=dec_params['n_quant'])

        self.encoder = enc.Encoder(n_in=self.preprocess.mfcc.n_out,
                parent_vc=self.preprocess.vc, **enc_params)

        bn_type = bn_params['type']
        bn_extra = dict((k, v) for k, v in bn_params.items() if k != 'type')
    
        # In each case, the objective function's 'forward' method takes the
        # same arguments.
        if bn_type == 'vqvae':
            self.bottleneck = vq_bn.VQ(**bn_extra, n_in=enc_params['n_out'])
            self.objective = vq_bn.VQLoss(self.bottleneck)

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
        self.decoder = dec.WaveNet(**dec_params, parent_vc=self.encoder.vc,
                n_lc_in=bn_params['n_out'])
        self.vc = self.decoder.vc
        #self.set_geometry()
        #self.set_slice_size(sam_per_slice)

    def __getstate__(self):
        state = { 
                'args': self.args,
                'state_dict': self.state_dict()
                }
        return state 

    def __setstate__(self, state):
        self.args = state['args']
        self._initialize()
        self.load_state_dict(state['state_dict'])

    def set_geometry(self):
        '''Compute the timestep offsets between the window boundaries of the
        encoder input wav, decoder input wav, and supervising wav input to the
        loss function'''
        self.vc.gen_stats(self.preprocess.vc)
        if self.bn_type in ('vae', 'vqvae'):
            self.objective.set_geometry(self.decoder.pre_upsample_vc,
                    self.decoder.last_grcc_vc)

        # timestep offsets between input and output of the encoder
        enc_off = vconv.offsets(self.preprocess.vc, self.decoder.last_upsample_vc)

        # timestep offsets between wav input and output of decoder 
        # NOTE: this starts from after the upsampling, because it is concerned
        # with the wav input, not conditioning vectors
        dec_off = vconv.offsets(self.decoder.last_upsample_vc.next(), self.decoder.vc)
        self.preprocess.set_geometry(enc_off, dec_off)

    def set_slice_size(self, n_sam_per_slice_req):
        self.vc.init_nv(n_sam_per_slice_req)
        self.input_size = self.preprocess.vc.src.nv 
        self.output_size = self.decoder.vc.dst.nv 

    def init_vq_embed(self, batch_gen):
        """Initialize the VQ Embedding with samples from the encoder."""
        if self.bn_type != 'vqvae':
            raise RuntimeError('init_vq_embed only applies to the vqvae model type')

        bn = self.bottleneck
        n = bn.emb.size()[0]
        e = 0
        
        while e != n:
            __, voice_inds_np, wav_np = next(batch_gen)
            __, mels, __, __ = self.preprocess(voice_inds_np, wav_np)
            encoding = self.encoder(mels)
            ze = self.bottleneck.linear(encoding)
            b = ze.size()[0]
            chunk = min(b, n - e)
            with torch.no_grad():
                bn.emb[e:e+chunk] = ze[0:chunk,:,0]
            e += chunk
        
    def checksum(self):
        '''Return checksum of entire set of model parameters'''
        return util.tensor_digest(self.parameters())
        
    def print_offsets(self):
        '''Show the set of offsets for each section of the model'''
        self.vc.print_chain()

    def forward(self, mels, wav_onehot_dec, voice_inds):
        '''
        B: n_batch
        T: receptive field of autoencoder
        T': receptive field of decoder 
        R: size of local conditioning output of encoder (T - encoder.vc.total())
        N: n_win (# consecutive samples processed in one batch channel)
        Q: n_quant
        wav_compand: (B, T)
        wav_onehot_dec: (B, T')  
        Outputs: 
        quant_pred (B, Q, N) # predicted wav amplitudes
        '''
        encoding = self.encoder(mels)
        encoding_bn = self.bottleneck(encoding)
        quant = self.decoder(wav_onehot_dec, encoding_bn, voice_inds)
        return quant

    def run(self, batch_gen):
        '''Run the model on one batch, returning the predicted and
        actual output
        B, T, Q: n_batch, n_timesteps, n_quant
        Outputs:
        quant_pred: (B, Q, T) (the prediction from the model)
        wav_compand_out: (B, T) (the actual data from the same timesteps)
        '''
        __, voice_inds_np, wav_np = next(batch_gen)
        voice_inds, mels, wav_onehot_dec, wav_compand_out = \
                self.preprocess(voice_inds_np, wav_np)

        quant = self.forward(mels, wav_onehot_dec, voice_inds)
        # quant_pred[:,:,0] is a prediction for wav_compand_out[:,1] 
        return quant[...,:-1], wav_compand_out[:,1:]

class Metrics(object):
    '''Manage running the model and saving output and target state'''
    def __init__(self, state):
        self.state = state
        self.quant = None
        self.target = None
        self.softmax = torch.nn.Softmax(1) # input to this is (B, Q, N)

    def update(self, batch_gen):
        # __, voice_inds_np, wav_np = next(batch_gen)
        quant_pred_snip, wav_compand_out_snip = self.state.model.run(batch_gen) 
        self.quant = quant_pred_snip
        self.target = wav_compand_out_snip
        self.probs = self.softmax(self.quant)

    def loss(self):
        '''This is the closure needed for the optimizer'''
        if self.quant is None or self.target is None:
            raise RuntimeError('Must call update() first')
        self.state.optim.zero_grad()
        loss = self.state.model.objective(self.quant, self.target)
        loss.backward()
        return loss
    
    def peak_dist(self):
        '''Average distance between the indices of the peaks in pred and
        target'''
        diffs = torch.argmax(self.quant, dim=1) - self.target 
        mean = torch.mean(torch.abs(diffs).float())
        return mean

    def avg_max(self):
        '''Average max value for the predictions.  As the prediction becomes
        more peaked, this should go up...'''
        max_val, max_ind = torch.max(self.probs, dim=1)
        mean = torch.mean(max_val)
        return mean
        
    def avg_prob_target(self):
        '''Average probability given to target'''
        target_probs = torch.gather(self.probs, 1, self.target.unsqueeze(1)) 
        mean = torch.mean(target_probs)
        return mean

