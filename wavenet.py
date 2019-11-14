import torch
from torch import nn
from torch import distributions as dist
import vconv
from numpy import prod as np_prod
import util
import netmisc

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

        
    def cond_lead(self):
        """
        distance from start of the overall stack input to
        the start of this convolution
        """
        l_off, r_off = vconv.output_offsets(self.wavenet_vc['beg_grcc'],
                self.vc)
        assert r_off == 0
        return l_off 

    def skip_lead(self):
        """
        distance from start of this *output* to start of the final stack
        output.  Note that the skip information is the *output* of self.vc, not
        the input.
        """
        if self.vc == self.wavenet_vc['end_grcc']:
            return 0

        l_off, r_off = vconv.output_offsets(self.vc.child,
                self.wavenet_vc['end_grcc'])
        assert r_off == 0
        return l_off 

    def forward(self, x, cond):
        """
        B, T: batchsize, win_size (determined from input)
        C, R, D, S: n_cond, n_res, n_dil, n_skp
        x: (B, R, T) (necessary shape for Conv1d)
        cond: (B, C, T) (necessary shape for Conv1d)
        returns: sig: (B, R, T), skp: (B, S, T) 
        """
        cond_lead = self.cond_lead()
        skip_lead = self.skip_lead()

        filt = self.conv_signal(x) + self.proj_signal(cond[:,:,cond_lead:])
        gate = self.conv_gate(x) + self.proj_gate(cond[:,:,cond_lead:])
        z = torch.tanh(filt) * torch.sigmoid(gate)
        sig = self.dil_res(z)
        skp = self.dil_skp(z[:,:,skip_lead:])
        sig += x[:,:,self.vc.l_wing_sz:]
        return sig, skp 


class Jitter(nn.Module):
    """Time-jitter regularization.  With probability [p, (1-2p), p], replace
    element i with element [i-1, i, i+1] respectively.  Disallow a run of 3
    identical elements in the output.  Let p = replacement probability, s =
    "stay probability" = (1-2p).
    
    tmp[i][j] = Categorical(a, b, c)
    encodes P(x_t|x_(t-1), x_(t-2)) 
    a 2nd-order Markov chain which generates a sequence in alphabet {0, 1, 2}. 
    
    The following meanings hold:

    0: replace element with previous
    1: do not replace 
    2: replace element with following

    For instance, suppose you have:
    source sequence: ABCDEFGHIJKLM
    jitter sequence: 0112021012210
    output sequence: *BCEDGGGIKLLL

    The only triplet that is disallowed is 012, which causes use of the same source
    element three times in a row.  So, P(x_t=0|x_(t-2)=2, x_(t-1)=1) = 0 and is
    renormalized.  Otherwise, all conditional distributions have the same shape,
    [p, (1-2p), p].

    Jitter has a "receptive field" of 3, and it is unpadded.  Our index mask will be
    pre-constructed to have {0, ..., n_win

    """
    def __init__(self, replace_prob):
        """n_win gives number of 
        """
        super(Jitter, self).__init__()
        p, s = replace_prob, (1 - 2 * replace_prob)
        self.register_buffer('probs', torch.tensor([p,s,p]).repeat(3,3,1))
        self.probs[2,1,:] = torch.tensor([0, s/(p+s), p/(p+s)])
        self.mindex = None
        self.adjust = None

    def gen_mask(self):
        """
        populates a tensor mask to be used for jitter, and sends it to GPU for
        next window
        """
        n_batch = self.mindex.shape[0]
        n_time = self.mindex.shape[1] - 1
        self.mindex[:,0:2] = 1
        for b in range(n_batch):
            # The Markov sampling process
            for t in range(2, n_time):
                p2, p1 = self.mindex[b,t-2], self.mindex[b,t-1]
                print(self.probs[p2,p1,:].shape)
                self.mindex[b,t] = \
                dist.Categorical(self.probs[p2,p1,:]).sample() 
            self.mindex[b, n_time] = 1

        # adjusts so that temporary value of mindex[i] = {0, 1, 2} imply {i-1,
        # i, i+1} also, first and last elements of mindex mean 'do not replace
        # the element with previous or next, but choose the existing element.
        # This prevents attempting to replace the first element of the input
        # with a non-existent 'previous' element, and likewise with the last
        # element.
        self.mindex += self.adjust 


    # Will this play well with back-prop?
    def forward(self, x):
        """
        Input: (B, I, T)
        """
        n_batch = x.shape[0]
        if self.mindex is None:
            n_time = x.shape[2]
            self.mindex = x.new_empty(n_batch, n_time + 1, dtype=torch.long)
            self.adjust = torch.arange(n_time + 1, dtype=torch.long,
                    device=x.device).repeat(n_batch, 1) - 2

        self.gen_mask()

        assert x.shape[2] == self.mindex.shape[1] - 1
        y = x.new_empty(x.shape)
        for b in range(n_batch):
            y[b] = torch.index_select(x[b], 1, self.mindex[b,1:])
        return y 


class Conditioning(nn.Module):
    """
    Module for merging up-sampled local conditioning vectors
    with voice ids.
    """
    def __init__(self, n_speakers, n_embed, bias=True):
        super(Conditioning, self).__init__()
        # Look at nn.embedding
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
        assert speaker_inds.dtype == torch.long
        # one_hot: (B, S)
        one_hot = util.gather_md(self.eye, 0, speaker_inds).permute(1, 0) 
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
    def __init__(self, *args, **kwargs):
        super(Conv1dWrap, self).__init__(*args, **kwargs)
        self.apply(netmisc.xavier_init)



class WaveNet(nn.Module):
    def __init__(self, filter_sz, n_lc_in, n_lc_out, lc_upsample_filt_sizes,
            lc_upsample_strides, n_res, n_dil, n_skp, n_post, n_quant,
            n_blocks, n_block_layers, jitter_prob, n_speakers, n_global_embed,
            bias=True, parent_vc=None):
        super(WaveNet, self).__init__()

        self.n_blocks = n_blocks
        self.n_block_layers = n_block_layers
        self.n_quant = n_quant
        self.quant_onehot = None 
        self.bias = bias
        self.jitter = Jitter(jitter_prob)
        post_jitter_filt_sz = 3
        lc_input_stepsize = np_prod(lc_upsample_strides) 

        lc_conv_name = 'LC_Conv(filter_size={})'.format(post_jitter_filt_sz) 
        self.lc_conv = Conv1dWrap(n_lc_in, n_lc_out,
                kernel_size=post_jitter_filt_sz, stride=1, bias=self.bias)

        cur_vc = vconv.VirtualConv(filter_info=post_jitter_filt_sz,
                stride=1, parent=parent_vc, name=lc_conv_name)

        self.vc = dict()
        self.vc['beg'] = cur_vc
        
        # This RF is the first processing of the local conditioning after the
        # Jitter. It is the starting point for the commitment loss aggregation
        self.vc['pre_upsample'] = cur_vc
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
        self.base_layer = Conv1dWrap(n_quant, n_res, kernel_size=1, stride=1,
                dilation=1, bias=self.bias)

        self.conv_layers = nn.ModuleList() 
        n_cond = n_lc_out + n_global_embed

        for b in range(self.n_blocks):
            for bl in range(self.n_block_layers):
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
        self.post1 = Conv1dWrap(n_skp, n_post, 1, bias=bias)
        self.post2 = Conv1dWrap(n_post, n_quant, 1, bias=bias)
        self.logsoftmax = nn.LogSoftmax(1) # (B, Q, N)
        self.vc['main'] = cur_vc

    def forward(self, wav_onehot, lc_sparse, speaker_inds, lcond_slice):
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
        lc_sparse = self.jitter(lc_sparse)
        lc_sparse = self.lc_conv(lc_sparse) 
        lc_dense = self.lc_upsample(lc_sparse)

        # Trimming due to different phases of the input MFCC windows
        trim_len = lcond_slice[0][1] - lcond_slice[0][0]
        lcd_sz = lc_dense.size()
        lc_dense_trim = lc_dense.new_empty(lcd_sz[0], lcd_sz[1], trim_len) 

        for b in range(lcd_sz[0]):
            sl_b, sl_e = lcond_slice[b]
            assert sl_e - sl_b == wav_onehot.shape[2]
            lc_dense_trim[b] = lc_dense[b,:,sl_b:sl_e]

        cond = self.cond(lc_dense_trim, speaker_inds)
        # "The conditioning signal was passed separately into each layer" - p 5 pp 1.
        # Oddly, they claim the global signal is just passed in as one-hot vectors.
        # But, this means wavenet's parameters would have N_s baked in, and wouldn't
        # be able to operate with a new speaker ID.

        sig = self.base_layer(wav_onehot) 
        skp_sum = None
        for i, l in enumerate(self.conv_layers):
            sig, skp = l(sig, cond)
            if skp_sum is None:
                skp_sum = skp
            else:
                skp_sum += skp
            
        post1 = self.post1(self.relu(skp_sum))
        quant = self.post2(self.relu(post1))
        # we only need this for inference time
        # logits = self.logsoftmax(quant) 
        return quant

