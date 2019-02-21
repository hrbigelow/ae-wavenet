import torch
import wave_encoder as we
import mel_mfcc as mm

import ae_model as ae


sample_rate = 16000 # timestep / second
sample_rate_ms = int(sample_rate / 1000) # timestep / ms 
window_length_ms = 25 # ms
hop_length_ms = 10 # ms
n_mels = 80
n_mfcc = 13



# preprocess
sample_file = '/home/henry/ai/data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac'
mp = mm.MfccProcess()
mda = mp.func(sample_file)

# encoder
kernel_size = 3
in_chan = mda.shape[0]
mid_chan = 768
enc = we.Encoder(in_chan, mid_chan)

print('mda.shape = ', mda.shape)


mda_ten = torch.tensor(mda, dtype=torch.float32)
mda_ten = mda_ten.unsqueeze(0)

latents = enc.forward(mda_ten)

print('latents.shape = ', latents.shape)

# bottleneck
bn_chan = 64
vae = bn.VAE(mid_chan, bn_chan, bias=False)

vae_samples = vae.forward

def main():
    args = get_args()

    # Construct model
    model = ae.AutoEncoder()

    # Restore from checkpoint
    # Initialize optimizer

    # Start training


if __name__ == '__main__':
    main()


