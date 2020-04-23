from torch import nn
import torch
import vconv
import parse_tools  
import wavenet as wn 

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
        super(MfccInverter, self).__init__()
        dec_params = self.init_args['dec_params']
        mi_params = self.init_args['mi_params']
        self.preprocess = wn.PreProcess(n_quant=dec_params['n_quant'])  
        self.wavenet = wn.WaveNet(**dec_params, parent_vc=None,
                n_lc_in=mi_params['n_lc_in'])

    def post_init(self, dataset):
        self._init_geometry(dataset.window_batch_size)

    def _init_geometry(self, batch_win_size):
        w = batch_win_size
        beg_grcc_vc = self.wavenet.vc['beg_grcc']
        end_grcc_vc = self.wavenet.vc['end_grcc']
        do = vconv.GridRange((0, 100000), (0, w), 1)
        di = vconv.input_range(beg_grcc_vc, end_grcc_vc, do)
        self.trim_dec_out = torch.tensor(
                [do.sub[0] - di.sub[0], do.sub[1] - di.sub[0]],
                dtype=torch.long)


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

    def forward(self, mels, wav_onehot_dec):
        """
        """
        quant = self.wavenet(wav_onehot_dec, mels, None, None)
        return quant

    def run(self, vbatch):
        """
        """
        wav_onehot_dec = self.preprocess(vbatch.wav_dec_input)
        trim = self.trim_dec_out
        wav_batch_out = vbatch.wav_dec_input[:,trim[0]:trim[1]]
        self.wav_onehot_dec = wav_onehot_dec
        quant = self.forward(vbatch.mel_enc_input, wav_onehot_dec)
        return quant[...,:-1], wav_batch_out[...,1:]

    
