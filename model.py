# Full Autoencoder model
import mfcc
import wave_encoder as enc
import vq_bn
import vae_bn
import ae_bn
import wavenet as dec 
import util
import torch
from torch import nn
from torch.nn.modules import loss
import rfield
import numpy as np


# from numpy import vectorize as np_vectorize
class PreProcess(nn.Module):
    '''Shape tensors by appropriate offsets to feed to Loss function'''
    def __init__(self, pre_params, n_quant):
        super(PreProcess, self).__init__()
        self.mfcc = mfcc.ProcessWav(**pre_params, name='mfcc')
        self.rf = self.mfcc.rf
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
                parent_rf=self.preprocess.rf, **enc_params)

        bn_type = bn_params['type']
        bn_extra = dict((k, v) for k, v in bn_params.items() if k != 'type')
    
        # In each case, the objective function's 'forward' method takes the
        # same arguments.
        if bn_type == 'vqvae':
            self.bottleneck = vq_bn.VQVAE(**bn_extra, n_in=enc_params['n_out'])
            self.objective = vq_bn.VQLoss(self.bottleneck, bn_params['beta'])

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
        self.decoder = dec.WaveNet(**dec_params, parent_rf=self.encoder.rf,
                n_lc_in=bn_params['n_out'])
        self.rf = self.decoder.rf
        self.set_geometry()
        self.set_slice_size(sam_per_slice)

    def __getstate__(self):
        state = { 
                'args': self.args,
                'state_dict': self.state_dict(),
                'rand_state': torch.default_generator.get_state()
                }
        return state 

    def __setstate__(self, state):
        self.args = state['args']
        self._initialize()
        self.load_state_dict(state['state_dict'])
        if 'rand_state' in state:
            torch.default_generator.set_state(state['rand_state'])

    def set_geometry(self):
        '''Compute the timestep offsets between the window boundaries of the
        encoder input wav, decoder input wav, and supervising wav input to the
        loss function'''
        self.rf.gen_stats(self.preprocess.rf)
        if self.bn_type in ('vae', 'vqvae'):
            self.objective.set_geometry(self.decoder.pre_upsample_rf,
                    self.decoder.last_grcc_rf)

        # timestep offsets between input and output of the encoder
        enc_off = rfield.offsets(self.preprocess.rf, self.decoder.last_upsample_rf)

        # timestep offsets between wav input and output of decoder 
        # NOTE: this starts from after the upsampling, because it is concerned
        # with the wav input, not conditioning vectors
        dec_off = rfield.offsets(self.decoder.last_upsample_rf.next(), self.decoder.rf)
        self.preprocess.set_geometry(enc_off, dec_off)

    def set_slice_size(self, n_sam_per_slice_req):
        self.rf.init_nv(n_sam_per_slice_req)
        self.input_size = self.preprocess.rf.src.nv 
        self.output_size = self.decoder.rf.dst.nv 

        
    def print_offsets(self):
        '''Show the set of offsets for each section of the model'''
        self.rf.print_chain()

    def forward(self, mels, wav_onehot_dec, voice_inds):
        '''
        B: n_batch
        T: receptive field of autoencoder
        T': receptive field of decoder 
        R: size of local conditioning output of encoder (T - encoder.rf.total())
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
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim
        self.quant = None
        self.target = None
        self.softmax = torch.nn.Softmax(1) # input to this is (B, Q, N)

    def update(self, batch_gen):
        __, voice_inds_np, wav_np = next(batch_gen)
        quant_pred_snip, wav_compand_out_snip = self.model.run(batch_gen) 
        self.quant = quant_pred_snip
        self.target = wav_compand_out_snip
        self.probs = self.softmax(self.quant)

    def loss(self):
        '''This is the closure needed for the optimizer'''
        if self.quant is None or self.target is None:
            raise RuntimeError('Must call update() first')
        self.optim.zero_grad()
        loss = self.model.objective(self.quant, self.target)
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

