# Full Autoencoder model
import wave_encoder as enc
import bottlenecks as bn
import wavenet as dec 
import util
import torch
from torch import nn
from torch.nn.modules import loss
import rfield as rf

class AutoEncoder(nn.Module):
    '''
    Full Autoencoder model
    '''
    def __init__(self, encoder_params, bn_params, decoder_params):
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
        decoder_params['n_in'] = bn_params['n_out']
        self.decoder = dec.WaveNet(**decoder_params)

        # Does the complete model need the loss function defined as well?
        self.loss = loss.CrossEntropyLoss() 

        # Offsets from encoder wav input and corresponding decoder wav input.
        # The decoder takes a much smaller wav input.
        ae_loff = self.encoder.foff.left + self.decoder.lc_foff.left
        ae_roff = self.encoder.foff.right + self.decoder.lc_foff.right
        self.ae_foff = rf.FieldOffset(offsets=(ae_loff, ae_roff))

        self.ckpt_path = util.CheckpointPath()

    def receptive_field_size(self):
        '''number of audio timesteps needed for a single output timestep prediction'''
        return self.ae_foff.total() + self.decoder.stack_foff.total()


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
            else:
                print('Warning: unknown module instance: {}'.format(str(type(m))))


    def forward(self, wav_enc, voice_ids):
        '''
        B, T: n_batch, n_win + recep_field_sz - 1
        T': subset of T, trimmed by self.ae_{loff/roff}
        Q: n_quant
        wav_enc: (B, T)
        wav_dec: (B, T') 
        outputs: (B, T, Q)  
        '''
        enc = self.encoder(wav_enc)
        enc_bn = self.bottleneck(enc)
        wav_dec = wav_enc[:,self.ae_foff.left:-self.ae_foff.right or None]
        quant = self.decoder(wav_dec, enc_bn, voice_ids)
        return quant 

    def loss_factory(self, batch_gen):
        _xent_loss = torch.nn.CrossEntropyLoss()
        pred_off = self.ae_foff.left + self.decoder.stack_foff.left + 1 
        def loss():
            ids, wav = next(batch_gen)
            quant = self.forward(wav, ids)
            return _xent_loss(quant, wav[:,pred_off:-self.ae_foff.right or None])
        return loss


