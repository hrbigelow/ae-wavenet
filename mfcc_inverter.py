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
        self.bn_type = None

        self.preprocess = wn.PreProcess(n_quant=dec_params['n_quant'])  
        self.wavenet = wn.WaveNet(**dec_params, parent_vc=None,
                n_lc_in=mi_params['n_lc_in'])
        self.objective = wn.RecLoss()

    def post_init(self, dataset):
        self.wavenet.set_parent_vc(dataset.mfcc_vc)
        self._init_geometry(dataset.window_batch_size)

    def _init_geometry(self, batch_win_size):
        w = batch_win_size
        mfcc_vc = self.wavenet.vc['beg'].parent
        beg_grcc_vc = self.wavenet.vc['beg_grcc']
        end_grcc_vc = self.wavenet.vc['end_grcc']
        end_ups_vc = self.wavenet.vc['last_upsample']

        # (d: decoder, m: mfcc, u: upsample), (o: output, i: input)
        do = vconv.GridRange((0, 100000), (0, w), 1)
        di = vconv.input_range(beg_grcc_vc, end_grcc_vc, do)
        wi = vconv.input_range(mfcc_vc, end_grcc_vc, do)
        mi = vconv.input_range(mfcc_vc.child, end_grcc_vc, do)
        uo = vconv.output_range(mfcc_vc.child, end_ups_vc, mi)

        self.enc_in_len = wi.sub_length()
        self.enc_in_mel_len = mi.sub_length()
        self.embed_len = mi.sub_length()
        self.dec_in_len = di.sub_length()

        # trims wav_enc_input to wav_dec_input
        self.trim_dec_in = torch.tensor(
                [di.sub[0] - wi.sub[0], di.sub[1] - wi.sub[0] ],
                dtype=torch.long)

        # needed by wavenet to trim upsampled local conditioning tensor
        self.wavenet.trim_ups_out = torch.tensor([di.sub[0] - uo.sub[0],
            di.sub[1] - uo.sub[0]], dtype=torch.long)

        self.trim_dec_out = torch.tensor(
                [do.sub[0] - di.sub[0], do.sub[1] - di.sub[0]],
                dtype=torch.long)
        self.wavenet.post_init()

    def print_geometry(self):
        """
        Print the convolutional geometry
        """
        vc = self.wavenet.vc['beg'].parent
        while vc is not None:
            print(vc)
            vc = vc.child

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

    def forward(self, mels, wav_onehot_dec, voice_inds, jitter_index):
        """
        """
        quant = self.wavenet(wav_onehot_dec, mels, voice_inds, jitter_index)
        return quant

    def run(self, vbatch):
        """
        """
        wav_onehot_dec = self.preprocess(vbatch.wav_dec_input)
        trim = self.trim_dec_out
        wav_batch_out = vbatch.wav_dec_input[:,trim[0]:trim[1]]
        self.wav_onehot_dec = wav_onehot_dec
        quant = self.forward(vbatch.mel_enc_input, wav_onehot_dec,
                vbatch.voice_index, vbatch.jitter_index)
        return quant[...,:-1], wav_batch_out[...,1:]

    
