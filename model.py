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

    def get_receptive_bounds(self):
        '''Calculate encoder and decoder input bounds relative to
        an output position at zero'''
        dec_beg = -self.decoder.foff.left
        dec_end = self.decoder.foff.right
        enc_beg = dec_beg - self.encoder.foff.left
        enc_end = dec_end + self.encoder.foff.right 
        return (enc_beg, enc_end), (dec_beg, dec_end) 



    def forward(self, wav, voice_ids):
        enc = self.encoder(wav)
        enc_bn = self.bottleneck(enc)
        logits = self.decoder(wav, enc_bn, voice_ids)
        return logits

