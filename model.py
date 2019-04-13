# Full Autoencoder model
import mfcc
import wave_encoder as enc
import bottlenecks as bn
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

    def set_geometry(self, dec_off, pred_off):
        self.l_dec_off, self.r_dec_off = dec_off
        self.l_pred_off, self.r_pred_off = pred_off


    def one_hot(self, wav_compand):
        '''
        wav_compand: (B, T)
        B, Q, T: n_batch, n_quant, n_timesteps
        returns: (B, Q, T)
        '''
        return util.gather_md(self.quant_onehot, 0, wav_compand.long()).transpose(1, 2)

    def forward(self, inds_np, wav_np):
        '''Inputs:
        B, T, M, Q: n_batch, n_timesteps, n_mels, n_quant
        inds_np: (B) (numpy)
        wav_np: (B, T)
        Outputs:
        inds: (B) (torch.tensor on current device)
        mels: (B,  
        wav_onehot_dec: (B, Q, T)
        wav_compand_out: (B, T)

        '''
        mels_np = np.apply_along_axis(self.mfcc.func, axis=1, arr=wav_np)

        # First moving of tensors to the destination device
        mels = torch.tensor(mels_np, device=self.dummy_buf.device)
        wav = torch.tensor(wav_np, device=self.dummy_buf.device)
        inds = torch.tensor(inds_np, device=self.dummy_buf.device)

        wav_dec = wav[:,self.l_dec_off:self.r_dec_off or None]
        wav_compand_dec = util.mu_encode_torch(wav_dec, self.n_quant)
        wav_compand_out = wav_compand_dec[:, self.l_pred_off:self.r_pred_off or None]
        wav_onehot_dec = self.one_hot(wav_compand_dec)
        return inds, mels, wav_onehot_dec, wav_compand_out


class AutoEncoder(nn.Module):
    '''
    Full Autoencoder model
    '''
    def __init__(self, pre_params, enc_params, bn_params, dec_params):
        self.args = [pre_params, enc_params, bn_params, dec_params]
        self._initialize()

    def _initialize(self):
        super(AutoEncoder, self).__init__() 
        pre_params, enc_params, bn_params, dec_params = self.args

        # the "preprocessing"
        self.preprocess = PreProcess(pre_params, n_quant=dec_params['n_quant'])

        self.encoder = enc.Encoder(n_in=self.preprocess.mfcc.n_out,
                parent_rf=self.preprocess.rf, **enc_params)

        bn_type = bn_params['type']

        bn_extra = dict((k, v) for k, v in bn_params.items() if k != 'type')
    
        if bn_type == 'vqvae':
            self.bottleneck = bn.VQVAE(**bn_extra, n_in=enc_params['n_out'])
        elif bn_type == 'vae':
            self.bottleneck = bn.VAE(**bn_extra, n_in=enc_params['n_out'])
        elif bn_type == 'ae':
            self.bottleneck = bn.AE(**bn_extra, n_in=enc_params['n_out'])
        else:
            raise InvalidArgument 

        self.decoder = dec.WaveNet(**dec_params, parent_rf=self.encoder.rf,
                n_lc_in=bn_params['n_out'])

        self.rf = self.decoder.rf

    def __getstate__(self):
        state = { 'args': self.args, 'state_dict': self.state_dict() }
        return state 

    def __setstate__(self, state):
        self.args = state['args']
        self._initialize()
        self.load_state_dict(state['state_dict'])

    def set_geometry(self, n_sam_per_slice_req):
        '''Compute the relationship between the encoder input, decoder input,
        and input to the loss function'''
        self.rf.gen_stats(n_sam_per_slice_req)

        enc_input = self.preprocess.rf.src
        dec_input = self.decoder.last_upsample_rf.dst
        loss_input = self.rf.dst

        dec_off = rfield.offsets(enc_input, dec_input)
        pred_off = rfield.offsets(dec_input, loss_input)
        self.preprocess.set_geometry(dec_off, pred_off)

        self.input_size = enc_input.nv 
        self.output_size = loss_input.nv 
        
    def print_offsets(self):
        '''Show the set of offsets for each section of the model'''
        self.rf.print_chain()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.2)
            elif isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.2)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            # else:
                # print('Warning: unknown module instance: {}'.format(str(type(m))))

    def forward(self, mels, wav_onehot_trim, voice_inds):
        '''
        B: n_batch
        T: total receptive field size of complete autoencoder model
        R: size of local conditioning output of encoder (T - encoder.rf.total())
        N: n_win (# consecutive samples processed in one batch channel)
        Q: n_quant
        wav_compand: (B, T)
        wav_onehot_trim: (B, T') 
        Outputs: (B, Q, N)  
        '''
        encoding = self.encoder(mels)
        encoding_bn = self.bottleneck(encoding)
        quant_pred = self.decoder(wav_onehot_trim, encoding_bn, voice_inds)
        return quant_pred

    def run(self, batch_gen):
        '''Run the model on one batch, returning the predicted and
        actual output'''
        __, voice_inds_np, wav_np = next(batch_gen)
        voice_inds, mels, wav_onehot_dec, wav_compand_out = \
                self.preprocess(voice_inds_np, wav_np)

        quant_pred = self.forward(mels, wav_onehot_dec, voice_inds)
        # quant_pred[:,:,0] is a prediction for wav_compand_out[:,1] 
        return quant_pred[:,:,:-1], wav_compand_out[:,1:]

class Metrics(object):
    '''Manage running the model and saving output and target state'''
    def __init__(self, model):
        self.model = model
        self.pred = None
        self.target = None
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(1)

    def update(self, batch_gen):
        __, voice_inds_np, wav_np = next(batch_gen)
        quant_pred_snip, wav_compand_out_snip = self.model.run(batch_gen) 
        self.pred = quant_pred_snip
        self.target = wav_compand_out_snip
        self.probs = self.softmax(self.pred)

    def loss(self):
        '''This is the closure needed for the optimizer'''
        if self.pred is None or self.target is None:
            raise RuntimeError('Must call update() first')
        return self.loss_fn(self.pred, self.target)
    
    def peak_dist(self):
        '''Average distance between the indices of the peaks in pred and
        target'''
        diffs = torch.argmax(self.pred, dim=1) - self.target 
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
        target_probs = util.gather_md(self.probs, 1, self.target)
        mean = torch.mean(target_probs)
        return mean

