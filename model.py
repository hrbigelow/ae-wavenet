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

    def get_receptive_field(self):
        raise NotImplementedError

    def forward(self, wav, voice_ids):
        enc = self.encoder(wav)
        enc_bn = self.bottleneck(enc)
        logits = self.decoder(wav, enc_bn, voice_ids)
        return logits

