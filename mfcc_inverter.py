from torch import nn
import torch
import vconv
import parse_tools  
import wavenet as wn 
import data

class MfccInverter(nn.Module):
    """
    WaveNet model for inverting the wave to mfcc function.
    Autoregressively generates wave data using MFCC local conditioning vectors
    does not use global condition vectors
    """
    def __init__(self, opts, dataset):
        opts_dict = vars(opts)
        dec_params = parse_tools.get_prefixed_items(opts_dict, 'dec_')
        dec_params['n_speakers'] = dataset.num_speakers()
        mi_params = parse_tools.get_prefixed_items(opts_dict, 'mi_')

        self.init_args = { 
                'dec_params': dec_params,
                'mi_params': mi_params 
                }
        self._initialize()

    def _initialize(self):
        """
        May be called either by __init__ (if running in 'new' mode) or
        by __setstate__ (if running in 'resume' mode)
        """
        super(MfccInverter, self).__init__()
        dec_params = self.init_args['dec_params']
        mi_params = self.init_args['mi_params']
        self.bn_type = 'none' 

        self.wavenet = wn.WaveNet(**dec_params, parent_vc=None,
                n_lc_in=mi_params['n_lc_in'])

        self.objective = wn.RecLoss()


    def override(self, n_win_batch=None):
        """
        override values from checkpoints
        """
        if n_win_batch is not None:
            self.window_batch_size = n_win_batch

    def post_init(self, dataset):
        """
        further initializations needed in case we are training the model
        """
        self.wavenet.set_parent_vc(dataset.mfcc_vc)
        self._init_geometry(self.window_batch_size)
        # self.print_geometry()


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

        self.trim_dec_out = torch.tensor(
                [end_gr.sub[0] - di.sub[0], end_gr.sub[1] - di.sub[0]],
                dtype=torch.long)

        self.wavenet.trim_ups_out = torch.tensor([0, beg_grcc_vc.in_len()],
                dtype=torch.long)

        self.wavenet.post_init(n_win_batch)


    def print_geometry(self):
        vc = self.wavenet.vc['beg'].parent
        while vc:
            print(vc)
            vc = vc.child

        print('trim_dec_in: {}'.format(self.trim_dec_in))
        print('trim_dec_out: {}'.format(self.trim_dec_out))
        print('trim_ups_out: {}'.format(self.wavenet.trim_ups_out))


    def __getstate__(self):
        state = { 
                'init_args': self.init_args,
                # 'state_dict': self.state_dict()
                }
        return state 

    def __setstate__(self, state):
        self.init_args = state['init_args']
        self._initialize()
        # self.load_state_dict(state['state_dict'])

    def forward(self, batch):
        b = batch
        if self.training:
            assert isinstance(b, data.VirtualBatch)
            return self.wavenet(b.wav, b.mel, b.voice_idx, b.jitter_idx, b)
        else:
            assert isinstance(b, data.MfccBatch)
            with torch.no_grad():
                return self.wavenet(b.wav, b.mel, b.voice_idx, b.jitter_idx, b)


    def run(self, vbatch):
        """
        """
        trim = self.trim_dec_out
        wav_batch_out = vbatch.wav[:,trim[0]:trim[1]]
        quant = self.forward(vbatch)

        pred, target = quant[...,:-1], wav_batch_out[...,1:]

        loss = self.objective(pred, target)
        ag_inputs = (vbatch.mel)
        (mel_grad, ) = torch.autograd.grad(loss, ag_inputs, retain_graph=True)
        self.objective.metrics.update({
            'mel_grad_sd': mel_grad.std(),
            'mel_grad_mean': mel_grad.mean()
            })
        return pred, target, loss 

