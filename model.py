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


class AutoEncoder(nn.Module):
    '''
    Full Autoencoder model
    '''
    def __init__(self, preprocess_params, encoder_params, bn_params,
            decoder_params):
        super(AutoEncoder, self).__init__() 

        # the "preprocessing"
        self.preprocess = mfcc.ProcessWav(**preprocess_params, name='mfcc')

        self.encoder = enc.Encoder(n_in=self.preprocess.n_out,
                parent_rf=self.preprocess.rf, **encoder_params)

        bn_type = bn_params.pop('type')

        # connecting encoder to bottleneck
        bn_params['n_in'] = encoder_params['n_out']
    
        if bn_type == 'vqvae':
            self.bottleneck = bn.VQVAE(**bn_params)
        elif bn_type == 'vae':
            self.bottleneck = bn.VAE(**bn_params)
        elif bn_type == 'ae':
            self.bottleneck = bn.AE(**bn_params)
        else:
            raise InvalidArgument 

        # connecting bottleneck to decoder.
        decoder_params['n_lc_in'] = bn_params['n_out']
        decoder_params['parent_rf'] = self.encoder.rf

        self.decoder = dec.WaveNet(**decoder_params)

        self.rf = self.decoder.rf

    def set_geometry(self, n_sam_per_slice_req):
        '''Compute the relationship between the encoder input, decoder input,
        and input to the loss function'''
        self.rf.gen_stats(n_sam_per_slice_req)

        enc_input = self.preprocess.rf.src
        dec_input = self.decoder.last_upsample_rf.dst
        loss_input = self.rf.dst

        self.l_dec_off, self.r_dec_off = rfield.offsets(enc_input, dec_input)
        self.l_pred_off, self.r_pred_off = rfield.offsets(dec_input, loss_input)

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

    def forward(self, mels, wav_onehot_trim, voice_ids):
        '''
        B: n_batch
        T: total receptive field size of complete autoencoder model
        R: size of local conditioning output of encoder (T - encoder.rf.total())
        N: n_win (# consecutive samples processed in one batch channel)
        Q: n_quant
        wav_compand: (B, T)
        wav_onehot_trim: (B, T') 
        outputs: (B, N, Q)  
        '''
        encoding = self.encoder(mels)
        encoding_bn = self.bottleneck(encoding)
        quant = self.decoder(wav_onehot_trim, encoding_bn, voice_ids)
        return quant 

    def loss_factory(self, batch_gen, device):
        _xent_loss = torch.nn.CrossEntropyLoss()
        # ids_to_inds = np_vectorize(self.speaker_id_map.__getitem__)

        def loss():
            # data module returns numpy.ndarray
            ids, inds, wav_raw = next(batch_gen)

            mels = np.apply_along_axis(self.preprocess.func, axis=1,
                    arr=wav_raw)

            # Here is where we transfer to GPU if necessary
            mels_ten = torch.tensor(mels, device=device)
            wav_ten = torch.tensor(wav_raw, device=device)
            inds_ten = torch.tensor(inds, device=device)
            wav_dec_ten = wav_ten[:,self.l_dec_off:self.r_dec_off or None]
            wav_compand_dec_ten = util.mu_encode_torch(wav_dec_ten,
                    self.decoder.n_quant)

            wav_compand_pred = wav_compand_dec_ten[:, self.l_pred_off:self.r_pred_off or None]
            wav_onehot_dec = self.decoder.one_hot(wav_compand_dec_ten)

            assert wav_ten.device == wav_onehot_dec.device
            assert wav_ten.device == inds_ten.device

            quant = self.forward(mels_ten, wav_onehot_dec, inds_ten)

            # quant[:,:,0] is the prediction for wav_compand_pred[:,1]
            return _xent_loss(quant[:,:,:-1], wav_compand_pred[:,1:])
        return loss


