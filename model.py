# Full Autoencoder model
import wave_encoder as enc
import bottlenecks as bn
import wavenet as dec 
import util
import torch
from torch import nn
from torch.nn.modules import loss
import rfield as rf
from numpy import vectorize as np_vectorize

class AutoEncoder(nn.Module):
    '''
    Full Autoencoder model
    '''
    def __init__(self, encoder_params, bn_params, decoder_params, speaker_ids):
        super(AutoEncoder, self).__init__() 

        self.encoder = enc.Encoder(**encoder_params)
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

        self.speaker_id_map = dict((v,k) for k,v in enumerate(speaker_ids))

        self.decoder = dec.WaveNet(**decoder_params)

        # Does the complete model need the loss function defined as well?
        self.loss = loss.CrossEntropyLoss() 

        # Offsets from encoder wav input and corresponding decoder wav input.
        # The decoder takes a much smaller wav input.
        # self.ae_rf = rf.Rfield(offsets=(ae_loff, ae_roff))

        self.ckpt_path = util.CheckpointPath()
        self.rf = self.decoder.rf

    def input_geometry(self, n_out_elem_req):
        n_in_elem, n_out_elem_act, l_out_off, r_out_off, is_valid = \
                self.rf.geometry(n_out_elem_req)
        if not is_valid:
            raise RuntimeError('Geometry of model is not valid.  Please see rfield for '
                    'details.')
        return n_in_elem, n_out_elem_act, l_out_off, r_out_off

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

    def forward(self, wav_raw, wav_onehot_trim, voice_ids):
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
        encoding = self.encoder(wav_raw)
        encoding_bn = self.bottleneck(encoding)
        quant = self.decoder(wav_onehot_trim, encoding_bn, voice_ids)
        return quant 

    def loss_factory(self, batch_gen):
        _xent_loss = torch.nn.CrossEntropyLoss()
        # offset between input to the decoder and its prediction
        pred_off = self.decoder.stack_rf.left + 1 
        ids_to_inds = np_vectorize(self.speaker_id_map.__getitem__)
        __, l_lc_off, r_lc_off, is_valid = \
                self.decoder.upsample_rf.geometry() # !!!

        def loss():
            # numpy.ndarray
            ids, wav_raw = next(batch_gen)
            wav_raw_dec = wav_raw[:,l_lc_off:r_lc_off or None]
            wav_compand_dec = torch.tensor(util.mu_encode_np(wav_raw_dec, self.decoder.n_quant))
            wav_compand_pred = wav_compand_dec[:, pred_off:]
            wav_onehot_dec = self.decoder.one_hot(wav_compand_dec)
            speaker_inds_ten = torch.tensor(ids_to_inds(ids))

            quant = self.forward(wav_raw, wav_onehot_dec, speaker_inds_ten)
            return _xent_loss(quant, wav_compand_pred)
        return loss


