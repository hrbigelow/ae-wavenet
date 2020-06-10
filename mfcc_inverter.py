from torch import nn
import torch
import vconv
import parse_tools  
import wavenet as wn 
import data
import mfcc

class MfccInverter(nn.Module):
    """
    WaveNet model for inverting the wave to mfcc function.
    Autoregressively generates wave data using MFCC local conditioning vectors
    does not use global condition vectors
    """
    def __init__(self, hps):
        super(MfccInverter, self).__init__()
        self.bn_type = 'none' 
        self.mfcc = mfcc.ProcessWav(
                sample_rate=hps.sample_rate, win_sz=hps.mfcc_win_sz,
                hop_sz=hps.mfcc_hop_sz, n_mels=hps.n_mels, n_mfcc=hps.n_mfcc)

        mfcc_vc = vconv.VirtualConv(filter_info=hps.mfcc_win_sz,
                stride=hps.mfcc_hop_sz, parent=None, name='MFCC')

        self.wavenet = wn.WaveNet(hps, parent_vc=mfcc_vc)
        self.objective = wn.RecLoss()
        self._init_geometry(hps.n_win_batch)


    def override(self, n_win_batch=None):
        """
        override values from checkpoints
        """
        if n_win_batch is not None:
            self.window_batch_size = n_win_batch


    def _init_geometry(self, n_win_batch):
        end_gr = vconv.GridRange((0, 100000), (0, n_win_batch), 1)
        end_vc = self.wavenet.vc['end_grcc']
        end_gr_actual = vconv.compute_inputs(end_vc, end_gr)

        mfcc_vc = self.wavenet.vc['beg'].parent
        beg_grcc_vc = self.wavenet.vc['beg_grcc']

        self.enc_in_len = mfcc_vc.in_len()
        self.enc_in_mel_len = self.embed_len = mfcc_vc.child.in_len()
        self.dec_in_len = beg_grcc_vc.in_len()

        di = beg_grcc_vc.input_gr
        wi = mfcc_vc.input_gr

        self.trim_dec_in = torch.tensor(
                [di.sub[0] - wi.sub[0], di.sub[1] - wi.sub[0] ],
                dtype=torch.long)

        # subrange on the wav input which corresponds to the output
        self.trim_dec_out = torch.tensor(
                [end_gr.sub[0] - wi.sub[0], end_gr.sub[1] - wi.sub[0]],
                dtype=torch.long)

        self.wavenet.trim_ups_out = torch.tensor([0, beg_grcc_vc.in_len()],
                dtype=torch.long)

        self.wavenet.post_init(n_win_batch)

    def get_input_size(self, output_size):
        return self.wavenet.get_input_size(output_size)

    def print_geometry(self):
        vc = self.wavenet.vc['beg'].parent
        while vc:
            print(vc)
            vc = vc.child

        print('trim_dec_in: {}'.format(self.trim_dec_in))
        print('trim_dec_out: {}'.format(self.trim_dec_out))
        print('trim_ups_out: {}'.format(self.wavenet.trim_ups_out))


    def forward(self, wav, mel, voice, jitter):
        if self.training:
            return self.wavenet(wav, mel, voice, jitter)
        else:
            with torch.no_grad():
                return self.wavenet(wav, mel, voice, jitter)


    def run(self, *inputs):
        """
        """
        wav, mel, voice, jitter = inputs
        mel.requires_grad_(True)

        trim = self.trim_dec_out
        wav_batch_out = wav[:,trim[0]:trim[1]]
        quant = self.forward(*inputs)

        pred, target = quant[...,:-1], wav_batch_out[...,1:]

        loss = self.objective(pred, target)
        ag_inputs = (mel)
        (mel_grad, ) = torch.autograd.grad(loss, ag_inputs, retain_graph=True)
        self.objective.metrics.update({
            'mel_grad_sd': mel_grad.std(),
            'mel_grad_mean': mel_grad.mean()
            })
        return pred, target, loss 

