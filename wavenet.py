import torch
from torch import nn
from torch import distributions as dist
import rfield as rf

class GatedResidualCondConv(nn.Module):
    def __init__(self, n_cond, n_res, n_dil, n_skp, stride, dil, filter_sz=2, bias=True):
        '''
        filter_sz: # elements in the dilated kernels
        n_cond: # channels of local condition vectors
        n_res : # residual channels
        n_dil : # output channels for dilated kernel
        n_skp : # channels output to skip connections
        '''
        super(GatedResidualCondConv, self).__init__()
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
        recep_field = (filter_sz - 1) * dil + 1
        self.foff = rf.FieldOffset(offsets=(recep_field - 1, 0))

        # kernel size is 1 for conditioning vectors, but filter_sz * dil + 1
        # for dilated convolutions.  So, the output of proj_* is longer than
        # conv_* by self.lead.  This is accounted for during the summation
        # steps in forward()
        self.lead = (filter_sz - 1) * dil

    def forward(self, x, cond):
        '''
        B, T: batchsize, win_size (determined from input)
        C, R, D, S: n_cond, n_res, n_dil, n_skp
        x: (B, R, T) (necessary shape for Conv1d)
        cond: (B, C, T) (necessary shape for Conv1d)
        returns: sig: (B, R, T), skp: (B, S, T) 
        '''
        filt = self.conv_signal(x) + self.proj_signal(cond[:,:,self.lead:])
        gate = self.conv_gate(x) + self.proj_gate(cond[:,:,self.lead:])
        z = torch.tanh(filt) * torch.sigmoid(gate)
        sig = self.dil_res(z)
        skp = self.dil_skp(z)
        sig += x[:,:,self.lead:]
        return sig, skp 

class Jitter(nn.Module):
    '''Time-jitter regularization.  With probability [p, (1-2p), p], replace
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

    '''
    def __init__(self, replace_prob):
        '''n_win gives number of 
        '''
        super(Jitter, self).__init__()
        p, s = replace_prob, (1 - 2 * replace_prob)
        tmp = torch.Tensor([p, s, p]).repeat(3, 3, 1)
        tmp[2][1] = torch.Tensor([0, s/(p+s), p/(p+s)])
        self.cond2d = [ [ dist.Categorical(tmp[i][j]) for i in range(3)] for j in range(3) ]


    def gen_mask(self, n_batch, n_win):
        '''populates a tensor mask to be used for jitter, and sends it to GPU for
        next window'''
        self.mindex = torch.empty(n_batch, n_win + 2, dtype=torch.long)
        self.mindex[:,0:2] = 1
        for b in range(n_batch):
            # The Markov sampling process
            for t in range(2, n_win + 1):
                self.mindex[b,t] = \
                        self.cond2d[self.mindex[b,t-2]][self.mindex[b,t-1]].sample()
            self.mindex[b, n_win + 1] = 1

        # adjusts so that temporary value of mindex[i] = {0, 1, 2} => {i-1, i, i+1}
        self.mindex += torch.arange(n_win + 2, dtype=torch.long).repeat(n_batch, 1) - 1


    # Will this play well with back-prop?
    def forward(self, x):
        '''Input: (B, I, T)'''
        n_batch, n_win = x.shape[0], x.shape[2]
        self.update_mask(n_batch, n_win)

        assert x.shape[2] == self.mindex.shape[1]
        y = torch.empty(x.shape, dtype=x.dtype)
        for b in range(self.n_batch):
            y[b] = torch.index_select(x[b], 1, self.mindex[b])
        return y[:,:,1:-1]

def _gather_md(input, dim, index):
    '''Gathers each sub-tensor in input, as specified by
    the value of each element in index at dim.
    Collects subtensors and arranges them by index shape.
    returns: (X1, ..., X_(d-1), X_(d+1), ... , Xn, I1, ..., Ik)
    X is index, I is input
    '''
    x = torch.index_select(input, dim, index.flatten())
    ish = list(input.shape)
    xsh = list(index.shape)
    nsh = xsh + [ish[i] for i in range(len(ish)) if i != dim]
    return x.reshape(nsh) 


class Conditioning(nn.Module):
    '''Module for merging up-sampled local conditioning vectors
    with voice ids.
    '''
    def __init__(self, n_speakers, n_embed, bias=True):
        super(Conditioning, self).__init__()
        self.speaker_embedding = nn.Linear(n_speakers, n_embed, bias)
        self.eye = torch.eye(n_speakers)

    def forward(self, lc, ids):
        '''
        I, G, S: n_in_chan, n_embed_chan, n_speakers
        lc : (B, T, I)
        ids: (B, T)
        returns: (B, T, I+G)
        '''
        one_hot = _gather_md(self.eye, 0, ids) # one_hot: (B, T, S)
        gc = self.speaker_embedding(one_hot) # gc: (B, T, G)
        all_cond = torch.cat((lc, gc), dim=2) 
        return all_cond


class Upsampling(nn.Module):
    '''Computes a one-per-timestep conditioning vector from a
    less-frequent input.   
    '''
    def __init__(self, n_lc_chan, lc_upsample_filt_sizes, lc_upsample_strides):
        super(Upsampling, self).__init__()
        self.tconvs = nn.ModuleList() 
        self.wings = []
        
        for filt_sz, stride in zip(lc_upsample_filt_sizes, lc_upsample_strides):
            left_wing_sz = (filt_sz - 1) // 2
            right_wing_sz = (filt_sz - 1) - left_wing_sz
            # Recall Pytorch's padding semantics for transpose conv.
            tconv = nn.ConvTranspose1d(n_lc_chan, n_lc_chan, filt_sz, stride,
                    padding=left_wing_sz)
            self.tconvs.append(tconv)
            self.wings.append((left_wing_sz, right_wing_sz))

        n = len(lc_upsample_strides)
        # sub_strides[i] will be the stride of the output of layer i relative to
        # the base-level timestep.
        sub_strides = [1] * n
        for i in reversed(range(n - 1)):
            sub_strides[i] = sub_strides[i+1] * lc_upsample_strides[i+1]
        
        loff, roff = 0, 0
        for i in range(n):
            loff += self.wings[i][0] * sub_strides[i]
            roff += self.wings[i][1] * sub_strides[i]

        self.foff = rf.FieldOffset(offsets=(loff, roff))

    def forward(lc):
        '''B, T, S, C: batch_sz, timestep, less-frequent timesteps, input channels
        lc: (B, S, C)
        returns: (B, T, C)
        Here, we trim the output of each transpose convolution, in order to discard
        output units that don't have full coverage of the filter with the input.
        '''
        for tconv, (left_wing_sz, right_wing_sz) in zip(self.tconvs, self.wings):
            lc = tconv.forward(lc)[:,:,left_wing_sz:-right_wing_sz or None]

        return lc




class WaveNet(nn.Module):
    def __init__(self, n_in, filter_sz, n_lc_in, n_lc_out, lc_upsample_filt_sizes,
            lc_upsample_strides, n_res, n_dil, n_skp, n_post, n_quant,
            n_blocks, n_block_layers, jitter_prob, n_speakers, n_global_embed,
            bias=True):
        super(WaveNet, self).__init__()

        self.n_blocks = n_blocks
        self.n_block_layers = n_block_layers
        self.bias = bias

        self.jitter = Jitter(jitter_prob)
        self.lc_conv = nn.Conv1d(n_lc_in, n_lc_out, 3, 1, bias=self.bias)
        self.lc_upsample = Upsampling(n_lc_out, lc_upsample_filt_sizes, lc_upsample_strides)
        self.cond = Conditioning(n_speakers, n_global_embed)

        self.base_layer = nn.Conv1d(n_in, n_res, 1, 1, dilation=1, bias=self.bias)

        self.conv_layers = nn.ModuleList() 
        n_cond = n_lc_out + n_global_embed
        for b in range(self.n_blocks):
            for bl in range(self.n_block_layers):
                dil = bl**2
                self.conv_layers.append(
                        GatedResidualCondConv(n_cond, n_res, n_dil, n_skp, 1, dil, filter_sz))

        self.post1 = nn.Conv1d(n_skp, n_post, 1, 1, 1, bias)
        self.post2 = nn.Conv1d(n_post, n_quant, 1, 1, 1, bias)
        self.logsoftmax = nn.LogSoftmax(2) # (B, T, C)

        # Calculate receptive field offsets, separately for the convolutional
        # stack, and for the upsampling of local conditioning
        stack_loff = sum(m.foff.left for m in self.conv_layers.children())
        stack_roff = sum(m.foff.right for m in self.conv_layers.children())
        self.stack_foff = rf.FieldOffset(offsets=(stack_loff, stack_roff))

        lc_loff = self.lc_upsample.foff.left
        lc_roff = self.lc_upsample.foff.right
        self.lc_foff = rf.FieldOffset(offsets=(lc_loff, lc_roff))



    def forward(self, x, lc, voice_ids):
        ''' B, T, I, L, Q: n_batch, n_win, n_in, n_lc_in, n_quant
        x: (B, I, T)
        lc: (B, L, T)
        voice_ids: (B, T)
        outputs: (B, T, Q)
        '''
        lc = self.jitter(lc)
        lc = self.lc_conv(lc) 
        lc = self.lc_upsample(lc)
        cond = self.cond(lc, voice_ids)
        # "The conditioning signal was passed separately into each layer" - p 5 pp 1.
        # Oddly, they claim the global signal is just passed in as one-hot vectors.
        # But, this means wavenet's parameters would have N_s baked in, and wouldn't
        # be able to operate with a new voice ID.

        sig = self.base_layer(x) 
        skp_sum = None
        for i, l in enumerate(self.conv_layers):
            sig, skp = l(sig, cond)
            if skp_sum: skp_sum += skp
            else: skp_sum = skp
            
        post1 = self.post1(nn.ReLU(skp_sum))
        quant = self.post2(nn.ReLU(post1))
        # we only need this for inference time
        # logits = self.logsoftmax(quant) 

        # quant: (B, T, Q), Q = n_quant
        return quant 


