# Full Autoencoder model
import wave_encoder as enc
import bottleneck as bn
import wavenet as dec 
from torch.nn.modules import loss

class AutoEncoder(nn.Module):
    '''
    Full Autoencoder model
    '''
    def __init__(self, bottleneck_type, params):
        super().__init__(self) # Is this necessary?
        self.encoder = enc.Encoder(n_in, n_mod)
        if bottleneck_type == 'vqvae':
            self.bottleneck = bn.VQVAE(n_in, n_out, bias)
        elif bottleneck_type == 'vae':
            self.bottleneck = bn.VAE(n_in, n_out, bias)
        elif bottleneck_type == 'ae':
            self.bottleneck = bn.AE(n_in, n_out, bias)
        else
            raise Error()

        self.decoder = dec.WaveNet(n_batch)

        # Does the complete model need the loss function defined as well?
        self.loss = loss.CrossEntropyLoss() 

    def forward(self, wav, mfcc, lc, voice_ids):
        enc = self.encoder(mfcc)
        enc_bn = self.bottleneck(enc)
        logits = self.decoder(wav, enc_bn, voice_ids)
        return logits


