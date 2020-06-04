import torch
from torch import nn
import vconv
import numpy as np
from numpy import prod as np_prod
import util
import netmisc
from collections import namedtuple
import torch.nn.functional as F

# from numpy import vectorize as np_vectorize
class PreProcess(nn.Module):
    """
    Perform one-hot encoding
    """
    def __init__(self, n_quant):
        super(PreProcess, self).__init__()
        self.n_quant = n_quant
        self.register_buffer('quant_onehot', torch.eye(self.n_quant))
        self.register_buffer('quant_range', torch.arange(0,
            self.n_quant).float())

    def one_hot(self, wav_compand):
        """
        wav_compand: (B, T)
        B, Q, T: n_batch, n_quant, n_timesteps
        returns: (B, Q, T)
        """
        wav_compand_tmp = wav_compand.long()
        wav_one_hot = util.gather_md_jit(self.quant_onehot, 0, (1,0),
                wav_compand_tmp).permute(1,0,2)
        return wav_one_hot

    

    def forward(self, in_snd_slice):
        """
        Converts the input to a one-hot format
        """
        in_snd_slice_onehot = self.one_hot(in_snd_slice)
        return in_snd_slice_onehot




class GatedResidualCondConv(nn.Module):
    def __init__(self, wavenet_vc, n_cond, n_res, n_dil, n_skp, stride, dil,
            filter_sz=2, bias=True, parent_vc=None, name=None):
        """
        filter_sz: # elements in the dilated kernels
        n_cond: # channels of local condition vectors
        n_res : # residual channels
        n_dil : # output channels for dilated kernel
        n_skp : # channels output to skip connections
        """
        super(GatedResidualCondConv, self).__init__()
        self.wavenet_vc = wavenet_vc 
        self.conv_signal = nn.Conv1d(n_res, n_dil, filter_sz, dilation=dil, bias=bias)
        self.conv_gate = nn.Conv1d(n_res, n_dil, filter_sz, dilation=dil, bias=bias)
        self.proj_signal = nn.Conv1d(n_cond, n_dil, kernel_size=1, bias=False)
        self.proj_gate = nn.Conv1d(n_cond, n_dil, kernel_size=1, bias=False)
        self.dil_res = nn.Conv1d(n_dil, n_res, kernel_size=1, bias=False)
        self.dil_skp = nn.Conv1d(n_dil, n_skp, kernel_size=1, bias=False)

        # The dilated autoregressive convolution produces an output at the
        # right-most position of the receptive field.  (At the very end of a
        # stack of these, the output corresponds to the position just after
        # this, but within the stack of convolutions, outputs right-aligned.
        dil_filter_sz = (filter_sz - 1) * dil + 1
        self.vc = vconv.VirtualConv(filter_info=(dil_filter_sz - 1, 0),
                parent=parent_vc, name=name)
        self.apply(netmisc.xavier_init)

    def post_init(self):
        """
        Initialize offset tensors
        """
        self.register_buffer('leads', torch.empty(4, dtype=torch.long))
        self.init_leads()
        self.set_full()

    def init_leads(self):
        """
        Update skip_lead and cond_lead to reflect changed geometry
        or chunk size.  Call this after vconv.compute_inputs is called
        """
        cond_lead, r_off = vconv.output_offsets(self.wavenet_vc['beg_grcc'],
                self.vc)
        assert r_off == 0

        if self.vc == self.wavenet_vc['end_grcc']:
            skip_lead = 0
        else:
            skip_lead, r_off = vconv.output_offsets(self.vc.child,
                    self.wavenet_vc['end_grcc'])
            assert r_off == 0

        self.leads[0] = cond_lead
        self.leads[1] = skip_lead
        self.leads[2] = self.vc.l_wing_sz
        self.leads[3] = 0

        self.global_rf = self.vc.in_len()
        self.local_rf = self.vc.filter_size()


    def set_incremental(self):
        """
        Set skip_lead and cond_lead for incremental operation
        """
        self.cond = 3
        self.skip = 3
        self.lw = 2

    def set_full(self):
        self.cond = 0
        self.skip = 1
        self.lw = 2

    def forward(self, x, cond):
        """
        B, T: batchsize, win_size (determined from input)
        C, R, D, S: n_cond, n_res, n_dil, n_skp
        x: (B, R, T) (necessary shape for Conv1d)
        cond: (B, C, T) (necessary shape for Conv1d)
        returns: sig: (B, R, T), skp: (B, S, T) 
        """
        cl, sl, lw = self.leads[self.cond], self.leads[self.skip], self.leads[self.lw]
        filt = self.conv_signal(x) + self.proj_signal(cond[:,:,cl:])
        gate = self.conv_gate(x) + self.proj_gate(cond[:,:,cl:])
        z = torch.tanh(filt) * torch.sigmoid(gate)
        sig = self.dil_res(z)
        skp = self.dil_skp(z[:,:,sl:])
        sig += x[:,:,lw:]

        return sig, skp 


class Conditioning(nn.Module):
    """
    Module for merging up-sampled local conditioning vectors
    with voice ids.
    """
    def __init__(self, n_speakers, n_embed, bias=True):
        super(Conditioning, self).__init__()
        # Look at nn.embedding
        self.n_speakers = n_speakers
        self.speaker_embedding = nn.Linear(n_speakers, n_embed, bias)
        self.register_buffer('eye', torch.eye(n_speakers))
        self.apply(netmisc.xavier_init)

    def forward(self, lc, speaker_inds):
        """
        I, G, S: n_in_chan, n_embed_chan, n_speakers
        lc : (B, T, I)
        speaker_inds: (B)
        returns: (B, T, I+G)
        """
        # one_hot: (B, S)
        one_hot = F.one_hot(speaker_inds.long(), self.n_speakers).float()
        # one_hot2 = util.gather_md_jit(self.eye, 0, (1,0), speaker_inds).permute(1, 0) 
        gc = self.speaker_embedding(one_hot) # gc: (B, G)
        gc_rep = gc.unsqueeze(2).expand(-1, -1, lc.shape[2])
        all_cond = torch.cat((lc, gc_rep), dim=1) 
        return all_cond

class Upsampling(nn.Module):
    def __init__(self, n_chan, filter_sz, stride, parent_vc, bias=True, name=None):
        super(Upsampling, self).__init__()
        # See upsampling_notes.txt: padding = filter_sz - stride
        # and: left_offset = left_wing_sz - end_padding
        end_padding = stride - 1
        self.vc = vconv.VirtualConv(
                filter_info=filter_sz, stride=stride,
                padding=(end_padding, end_padding), is_downsample=False,
                parent=parent_vc, name=name
                )

        self.tconv = nn.ConvTranspose1d(n_chan, n_chan, filter_sz, stride,
                padding=filter_sz - stride, bias=bias)
        self.apply(netmisc.xavier_init)

    def forward(self, lc):
        """
        B, T, S, C: batch_sz, timestep, less-frequent timesteps, input channels
        lc: (B, C, S)
        returns: (B, C, T)
        """
        lc_up = self.tconv(lc)
        return lc_up

class Conv1dWrap(nn.Conv1d):
    """
    Simple wrapper that ensures initialization
    """
    def __init__(self, name, parent_vc, **kwargs):
        super(Conv1dWrap, self).__init__(**kwargs)
        self.apply(netmisc.xavier_init)
        self.vc = vconv.VirtualConv(filter_info=kwargs['kernel_size'],
                stride=kwargs['stride'],
                name=name, parent=parent_vc)


"""
Architecture parameters that remain constant across save/restore.
Note that n_batch and n_batch_win are excluded, since they may
vary across save/restore cycles
"""
ArchConfig = namedtuple('ArchConfig', [
    'n_lc_in',
    'n_lc_out',
    'n_res',
    'n_dil',
    'n_skp',
    'n_post',
    'n_quant',
    'n_blocks',
    'n_block_layers',
    'n_speakers',
    'n_global_embed'
    ]
    )

"""
class ArchConfig(object):
    def __init__(self, n_lc_in, n_lc_out, n_res, n_dil,
            n_skp, n_post, n_quant, n_blocks, n_block_layers, n_speakers,
            n_global_embed):
        self.n_lc_in = n_lc_in
        self.n_lc_out = n_lc_out
        self.n_res = n_res
        self.n_dil = n_dil
        self.n_skp = n_skp
        self.n_post = n_post
        self.n_quant = n_quant
        self.n_blocks = n_blocks
        self.n_block_layers = n_block_layers
        self.n_speakers = n_speakers,
        self.n_global_embed = n_global_embed
"""


class WaveNet(nn.Module):
    # see https://pytorch.org/docs/stable/jit_language_reference.html \\
    # #for-loops-over-constant-nn-modulelist
    __constants__ = ['conv_layers']

    def __init__(self, filter_sz, n_lc_in, n_lc_out, lc_upsample_filt_sizes,
            lc_upsample_strides, n_res, n_dil, n_skp, n_post, n_quant,
            n_blocks, n_block_layers, n_speakers, n_global_embed,
            bias=True, parent_vc=None):
        super(WaveNet, self).__init__()

        self.ac = ArchConfig(n_lc_in, n_lc_out, n_res,
                n_dil, n_skp, n_post, n_quant, n_blocks, n_block_layers,
                n_speakers, n_global_embed)
        self.n_blocks = n_blocks
        self.n_block_layers = n_block_layers
        self.n_skp = n_skp
        self.n_res = n_res
        self.n_quant = n_quant

        self.quant_onehot = None 
        self.bias = bias
        post_jitter_filt_sz = 3
        lc_input_stepsize = np_prod(lc_upsample_strides) 

        lc_conv_name = 'LC_Conv(filter_size={})'.format(post_jitter_filt_sz) 
        self.lc_conv = Conv1dWrap(lc_conv_name, parent_vc, in_channels=n_lc_in,
                out_channels=n_lc_out, kernel_size=post_jitter_filt_sz,
                stride=1, bias=self.bias)

        # cur_vc = vconv.VirtualConv(filter_info=post_jitter_filt_sz,
        #         stride=1, parent=parent_vc, name=lc_conv_name)

        self.vc = dict()
        self.vc['beg'] = self.lc_conv.vc 
        cur_vc = self.vc['beg']

        self.preprocess = PreProcess(self.ac.n_quant) 
        
        # This VC is the first processing of the local conditioning after the
        # Jitter. It is the starting point for the commitment loss aggregation
        self.lc_upsample = nn.Sequential()

        # WaveNet is a stand-alone model, so parent_vc is None
        # The Autoencoder model in model.py will link parent_vcs together.
        for i, (filt_sz, stride) in enumerate(zip(lc_upsample_filt_sizes,
            lc_upsample_strides)):
            name = 'Upsampling_{}(filter_sz={}, stride={})'.format(i, filt_sz, stride)   
            mod = Upsampling(n_lc_out, filt_sz, stride, cur_vc, name=name)
            self.lc_upsample.add_module(str(i), mod)
            cur_vc = mod.vc

        # This vc describes the bounds of the input wav corresponding to the
        # local conditioning vectors
        self.vc['last_upsample'] = cur_vc
        self.cond = Conditioning(n_speakers, n_global_embed)
        self.base_layer = Conv1dWrap('Base Layer', cur_vc, in_channels=n_quant,
                out_channels=n_res, kernel_size=1, stride=1, dilation=1,
                bias=self.bias)

        self.base_layer.vc.do_trim_input = True
        cur_vc = self.base_layer.vc

        self.conv_layers = nn.ModuleList() 
        n_cond = n_lc_out + n_global_embed

        for b in range(self.ac.n_blocks):
            for bl in range(self.ac.n_block_layers):
                dil = 2**bl
                name = 'GRCC_{},{}(dil={})'.format(b, bl, dil)
                grc = GatedResidualCondConv(self.vc, n_cond, n_res, n_dil,
                        n_skp, 1, dil, filter_sz, bias, cur_vc, name)
                self.conv_layers.append(grc)
                cur_vc = grc.vc


        # Each module in the stack needs to know the dimensions of
        # the input and output of the overall stack, in order to trim
        # residual connections
        self.vc['beg_grcc'] = self.conv_layers[0].vc
        self.vc['end_grcc'] = self.conv_layers[-1].vc 

        self.relu = nn.ReLU()
        self.post1 = Conv1dWrap('Post1', cur_vc, in_channels=n_skp,
                out_channels=n_post, kernel_size=1, stride=1, bias=bias)

        self.post2 = Conv1dWrap('Post2', self.post1.vc, in_channels=n_post,
                out_channels=n_quant, kernel_size=1, stride=1, bias=bias)
        self.logsoftmax = nn.LogSoftmax(1) # (B, Q, N)
        self.vc['main'] = self.post2.vc 

    def set_parent_vc(self, parent_vc):
        self.vc['beg'].parent = parent_vc
        parent_vc.child = self.vc['beg']


    def post_init(self, n_batch_win):
        self.n_batch_win = n_batch_win

        one_gr = vconv.GridRange((0, int(1e12)), (0, 1), 1)
        vconv.compute_inputs(self.vc['end_grcc'], one_gr)

        mfcc_vc = self.vc['beg'].parent
        beg_grcc_vc = self.vc['beg_grcc']
        self.wav_cond_offset = int(beg_grcc_vc.input_gr.sub[0] - mfcc_vc.input_gr.sub[0])

        for layer in self.conv_layers:
            layer.post_init()

        self.base_global_rf = self.conv_layers[0].global_rf

    def set_n_replicas(self, n_replicas):
        self.n_replicas = n_replicas

    def set_incremental(self):
        """
        Set cond_lead and skip_leads for incremental mode
        """
        for layer in self.conv_layers:
            layer.set_incremental()

    def set_full(self):
        """
        Set for full inference mode
        """
        for layer in self.conv_layers:
            layer.set_full()  


    def forward(self, wav, lc_sparse, speaker_inds, jitter_index):
        if self.training:
            return self.forward_train(wav, lc_sparse, speaker_inds,
                    jitter_index)
        else:
            return self.forward_test(wav, lc_sparse, speaker_inds,
                    jitter_index)


    def forward_train(self, wav, lc_sparse, speaker_inds, jitter_index):
        """
        B: n_batch (# of separate wav streams being processed)
        T1: n_wav_timesteps
        T2: n_conditioning_timesteps
        I: n_in
        L: n_lc_in
        Q: n_quant

        wav: (B, Q, T1)
        lc: (B, L, T2)
        speaker_inds: (B, T)
        outputs: (B, Q, N)
        """
        D1 = lc_sparse.size()[1]
        lc_jitter = torch.take(lc_sparse,
                jitter_index.unsqueeze(1).expand(-1, D1, -1))
        lc_conv = self.lc_conv(lc_jitter) 
        lc_dense = self.lc_upsample(lc_conv)

        # Trimming due to different phases of the input MFCC windows
        # W2 = lcond_slice.size()[1] 

        D2 = lc_dense.size()[1]
        lc_dense_trim = lc_dense[:,:,self.trim_ups_out[0]:self.trim_ups_out[1]]
        # lc_dense_trim = torch.take(lc_dense,
        #         lcond_slice.unsqueeze(1).expand(-1, D2, -1))

        cond = self.cond(lc_dense_trim, speaker_inds)
        # "The conditioning signal was passed separately into each layer" - p 5 pp 1.
        # Oddly, they claim the global signal is just passed in as one-hot vectors.
        # But, this means wavenet's parameters would have N_s baked in, and wouldn't
        # be able to operate with a new speaker ID.
        wav_onehot = F.one_hot(wav.long(), self.n_quant).permute(0,2,1)
        # wav_onehot = self.preprocess(wav)

        sig = self.base_layer(wav_onehot) 
        skp_sum = torch.zeros(wav_onehot.shape[0], 256,
                self.n_batch_win, device=wav_onehot.device)

        for layer in self.conv_layers:
            sig, skp = layer(sig, cond)
            skp_sum += skp
            
        post1 = self.post1(self.relu(skp_sum))
        quant = self.post2(self.relu(post1))

        # we only need this for inference time
        # logits = self.logsoftmax(quant) 
        return quant


    def forward_test(self, wav, lc_sparse, speaker_inds, jitter_index):
        """
        Generate n_rep samples, using lc_sparse and speaker_inds for local and global
        conditioning.  

        wav_onehot: full length wav vector
        lc_sparse: full length local conditioning vector derived from full
        wav_onehot
        """
        n_rep = torch.tensor(self.n_replicas, device=wav.device)

        # wav_onehot = self.preprocess(wav)

        # assert wav.min().item().toLong() >= 0
        wav_onehot = F.one_hot(wav.long(), self.n_quant).permute(0,2,1).float()

        # !!! this may be where a huge bug is
        wav_onehot = wav_onehot[:,:,self.wav_cond_offset:]

        # for converting from one-hot to value format
        # quant_range = wav_onehot.new(list(range(self.ac.n_quant)))

        lc_sparse = lc_sparse.repeat(n_rep, 1, 1)
        jitter_index = jitter_index.repeat(n_rep, 1)
        speaker_inds = speaker_inds.repeat(n_rep)

        # precalculate conditioning vector for all timesteps
        D1 = lc_sparse.size()[1]
        lc_jitter = torch.take(lc_sparse,
                jitter_index.unsqueeze(1).expand(-1, D1, -1))
        lc_conv = self.lc_conv(lc_jitter) 
        lc_dense = self.lc_upsample(lc_conv)
        cond = self.cond(lc_dense, speaker_inds)
        n_ts = cond.size()[2]

        chunk_size = 1000

        # first slot is to report the original 
        wav_onehot = wav_onehot.repeat(n_rep + 1, 1, 1)
        n_layers = self.n_blocks * self.n_block_layers
        # n_layers = len(self.conv_layers)

        # sig[0] is the output of the base_layer
        # sig[i] is the output of the conv_layer[i-1]
        # there is no sig to hold the output of conv_layer[-1]
        # instead, it is directed to sig[n_layers-1]

        # wav_irng slices wav_onehot when used as input, and
        # we derive the single position output from wav_irng
        irng = wav_onehot.new_empty(n_layers, 2, dtype=torch.long)

        # orng[l] is the output range of layer l, which populates sig[l]
        # except that orng[-2] and orng[-1] both populate sig[-1]
        # because
        orng = wav_onehot.new_empty(n_layers + 1, 2, dtype=torch.long)

        # the one  
        cond_rng = wav_onehot.new_empty(2, dtype=torch.long)

        # input range for the wav_onehot vector
        wav_ir = wav_onehot.new_empty(2, dtype=torch.long)

        skp_sum = torch.zeros(n_rep, self.n_skp, 1,
                device=wav_onehot.device)

        # forward-most index element in wave input
        cur_pos = torch.tensor([self.base_global_rf], dtype=torch.long,
                device=wav_onehot.device)
        # end_pos = torch.tensor([self.base_global_rf + 30000], dtype=torch.long,
        #         device=wav_onehot.device)
        end_pos = torch.tensor([n_ts], dtype=torch.long,
                device=wav_onehot.device)

        wav_ir[0] = cur_pos[0] - self.base_global_rf 
        wav_ir[1] = cur_pos[0]

        sig = []
        i = 0
        for l in self.conv_layers:
            # print(n_rep, self.n_res, l.global_rf, chunk_size, 1)
            sig.append(torch.empty(n_rep, self.n_res, l.global_rf + chunk_size -
                    1, device=wav_onehot.device))
            irng[i,0] = 0 
            irng[i,1] = l.global_rf 
            orng[i,0] = 0 
            orng[i,1] = l.global_rf
            i += 1

        orng[-1,0] = 0
        orng[-1,1] = 1
        cond_rng[0] = wav_ir[0] 
        cond_rng[1] = wav_ir[1]

        report_interval = torch.tensor(1000, dtype=torch.long,
                device=wav_onehot.device)
        zero = torch.tensor(0, dtype=torch.long, device=wav_onehot.device)

        self.set_full() 
        while not torch.equal(cur_pos, end_pos):
            chunk_size = min(chunk_size, end_pos[0] - cur_pos[0])

            for _ in range(chunk_size):
                # base_layer is a 1x1 convolution, so uses irng[0] 
                # for both input and output
                ir = irng[0]
                sig[0][:,:,ir[0]:ir[1]] = \
                        self.base_layer(wav_onehot[1:,:,wav_ir[0]:wav_ir[1]]) 
                skp_sum[...] = 0

                li = 0
                for layer in self.conv_layers:
                    # last iteration reassigns to same sig slot (unused) 
                    li_out = min(li+1, n_layers - 1)
                    p, q = irng[li], orng[li+1]
                    sig[li_out][:,:,q[0]:q[1]], skp = \
                            layer(sig[li][:,:,p[0]:p[1]], cond[:,:,cond_rng[0]:cond_rng[1]])
                    skp_sum += skp
                    li += 1

                post1 = self.post1(self.relu(skp_sum))
                quant = self.post2(self.relu(post1)).squeeze(2)
                probs = F.softmax(quant, dim=-1)
                indices = torch.multinomial(probs, 1, True) 
                wav_onehot[1:,:,wav_ir[1]] = F.one_hot(indices,
                        self.n_quant).squeeze(1).float()

                # post_val = int(torch.matmul(wav_onehot[0,:,wav_ir[1]],
                #     quant_range))
                # print('{}: {} - {}'.format(post_val - pre_val, pre_val,
                #     post_val))

                if torch.equal(irng[0,0], zero):
                    # finished initialization, now incremental mode
                    # we only really need 1 new element, but computing two
                    # nicely fits with sig[0]
                    self.set_incremental()
                    cond_rng[0] = cond_rng[1] - 1
                    # wav_ir[0] = wav_ir[1] - local_rf[0]

                    li = 0
                    for l in self.conv_layers:
                        # hack because we can't index self.conv_layers
                        if li == 0:
                            wav_ir[0] = wav_ir[1] - l.local_rf
                        irng[li,0] = l.global_rf - l.local_rf 
                        irng[li,1] = l.global_rf 
                        orng[li,0] = l.global_rf - 1 
                        orng[li,1] = l.global_rf
                        li += 1
                        
                orng += 1
                irng += 1
                wav_ir += 1
                cond_rng += 1
                cur_pos += 1

                if torch.equal(torch.fmod(wav_ir[1], report_interval), zero):
                # if wav_ir[1] % 1000 == 0:
                    print('On timestep {} out of {}'.format(wav_ir[1].item(),
                        end_pos[0].item()))

            # reset windows
            for i in range(n_layers):
                sig[i][:,:,:-chunk_size] = sig[i][:,:,chunk_size:]
            irng -= chunk_size
            orng -= chunk_size

        
        # convert to value format
        wav = torch.matmul(wav_onehot.permute(0,2,1), self.preprocess.quant_range)

        # print(wav[:,end_pos:end_pos + 10000])
        print('synth range std: {}, baseline std: {}'.format(
            wav[:,:end_pos[0]].std(), wav[:,end_pos[0]:].std()
            ))

        return wav
        # return wav[0,...], wav[1:,...]



class RecLoss(nn.Module):
    def __init__(self):
        super(RecLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(1) # input is (B, Q, N)

    def forward(self, quant_pred, target_wav):

        log_pred = self.logsoftmax(quant_pred)
        target_wav_gather = target_wav.long().unsqueeze(1)
        log_pred_target = torch.gather(log_pred, 1, target_wav_gather)

        rec_loss = - log_pred_target.mean()
        self.metrics = {
                'rec': rec_loss
                }

        return rec_loss 

