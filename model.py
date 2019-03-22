# Full Autoencoder model
import wave_encoder as enc
import bottlenecks as bn
import wavenet as dec 
from torch import nn
from torch.nn.modules import loss

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

        # offsets between the encoder window bounds and
        # corresponding decoder window bounds
        self.enc_dec_loff = self.decoder.foff.left - self.encoder.foff.left
        self.enc_dec_roff = self.encoder.foff.right - self.decoder.foff.right
        assert self.enc_dec_loff > 0
        assert self.enc_dec_roff > 0

    def get_receptive_bounds(self):
        '''Calculate encoder and decoder input bounds relative to
        an output position at zero.'''
        dec_beg = -self.decoder.foff.left
        dec_end = self.decoder.foff.right
        enc_beg = dec_beg - self.encoder.foff.left
        enc_end = dec_end + self.encoder.foff.right 
        return (enc_beg, enc_end), (dec_beg, dec_end) 


    def forward(self, wav_enc, wav_dec, voice_ids):
        '''
        B, T: n_batch, n_win + recep_field_sz - 1
        T': subset of T, trimmed by self.enc_dec_{loff/roff}
        Q: n_quant
        wav_enc: (B, T)
        wav_dec: (B, T') 
        outputs: (B, T, Q)  
        '''
        enc = self.encoder(wav_enc)
        enc_bn = self.bottleneck(enc)
        quant = self.decoder(wav_dec, enc_bn, voice_ids)
        return quant 

    def loss_factory(self, batch_gen):
        _xent_loss = torch.CrossEntropyLoss()
        def loss():
            ids, wavs_enc = next(batch_gen)
            wavs_dec = wavs_enc[:,self.enc_dec_loff:-self.enc_dec_roff]
            # quant[0] is a prediction for wavs_dec[1], etc
            # quant: (B, W, 
            quant = self.forward(wavs_enc, wavs_dec, ids)
            return _xent_loss(quant[:,:-1], wavs_dec[:,1:])
        return loss


            




