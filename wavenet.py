# Issues:  1. We do NOT want padded convolutions - we will assume that x and lc
# have the same time-step resolution, and the same dimension for batching.

# Regarding batching, I need to figure out what dimension that will be on
# as well, and how nn.Conv1d deals with it.

# Regarding training with multiple input windows, I need to figure out how to
# store autoregressive state so that successive window ranges are properly
# initialized

class GatedResidualCondConv(nn.Module):
    def __init__(self, n_lc, n_kern, n_res, n_dil, n_skp, stride, dil, bias=True):
        self.conv_signal = nn.Conv1d(n_res, n_dil, n_kern, stride, dil, bias)
        self.conv_gate = nn.Conv1d(n_res, n_dil, n_kern, stride, dil, bias)
        self.proj_signal = nn.Conv1d(n_lc, n_res, 1, bias=False)
        self.proj_gate = nn.Conv1d(n_lc, n_res, 1, bias=False)
        self.dil_res = nn.Conv1d(n_dil, n_res, 1, bias=False)
        self.dil_skp = nn.Conv1d(n_dil, n_skp, 1, bias=False)

    def forward(self, x, lc):
        filt = self.conv_signal(x) + self.proj_signal(lc)
        gate = self.conv_gate(x) + self.proj_gate(lc)
        z = torch.tanh(filt) * torch.sigmoid(gate)
        sig = self.dil_res(z)
        skp = self.dil_skp(z)
        sig += x
        return sig, skp 

class Jitter(nn.Module):
    '''Time-jitter regularization.  With probability [p, (1-2p), p], replace element i
    with element [i-1, i, i+1] respectively.  Disallow a run of 3 identical elements in
    the output.  Let p = replacement probability, s = "stay probability" = (1-2p).
    '''
    def __init__(self, n_batch, n_win, replace_prob):
        p, s = replace_prob, (1 - 2 * replace_prob)
        tmp = torch.tensor([p, s, p]).repeat(1, 3, 3)
        tmp[0][1] = torch.tensor([p/(p+s), s/(p+s), 0])
        self.cond2d = [ [ dist.Categorical(tmp[i][j]) for i in range(3)] for j in range(3) ]
        self.n_win = n_win
        self.mindex = torch.ones(n_batch, n_win)

    def update_mask(self):
        '''populates a tensor mask to be used for jitter, and sends it to GPU for
        next window'''
        for b in range(n_batch):
            for t in range(2, n_win):
                self.mindex = self.cond2d[self.mindex[b][t-2]][self.mindex[b][t-1]].sample()
            self.mindex += torch.arange(n_win) - 1

    def forward(self, x):
        '''Input: (B, T, I)'''
        return torch.index_select(x, 1, self.mindex) 



class WaveNet(nn.Module):
    def __init__(self, n_batch, n_in, n_kern, n_lc_in, n_lc_out, lc_upsample_strides,
            lc_upsample_kern_sizes,
            n_res, n_dil, n_skp, n_post, n_quant,
            n_blocks, n_block_layers, jitter_prob, bias=True):
        self.n_blocks = n_blocks
        self.n_block_layers = n_block_layers
        self.bias = bias
        self.jitter = Jitter(n_win, n_batch, jitter_prob)
        self.lc_conv = nn.Conv1d(n_lc_in, n_lc_out, 3, 1, bias=self.bias)
        # LC upsampling layers
        tmp_mods = []
        for kern_size, stride in zip(lc_upsample_kern_sizes, lc_upsample_strides):
            tmp_mods.append(nn.ConvTranspose1d(n_lc_out, n_lc_out, kern_size, stride))

        self.lc_upsample = nn.Sequential(*tmp_mods)
        self.one_hot_enc = 


        self.base_layer = nn.Conv1d(n_in, n_res, self.kern_size, self.stride,
                dilation=1, bias=self.bias)

        self.conv_layers = nn.ModuleList() 
        self.conv_state = []
        for b in range(self.n_blocks):
            for bl in range(self.n_block_layers):
                dil = bl**2
                self.conv_layers.append(
                        GatedResidualCondConv(n_lc, 2, n_res, n_dil, n_skp, 1, 1, dil))
                self.conv_state.append(torch.zeros([n_batch, n_res, dil], dtype=torch.float32))

        self.post1 = nn.Conv1d(n_skp, n_post, 1, 1, 1, bias)
        self.post2 = nn.Conv1d(n_post, n_quant, 1, 1, 1, bias)
        self.softmax = nn.Softmax(1) # Fix this dimension

    def forward(self, x, lc, voice_ids):
        ''' B = n_batch, T = n_win, I = n_in, L = n_lc_in
        x: (B, T, I)
        lc: (B, T, L)
        voice_ids: (B)
        '''
        lc = self.jitter(lc)
        lc = self.lc_conv(lc) 
        lc = self.lc_upsample(lc)

        all_cond = 

        sig = self.base_layer(x) 
        skp_sum = None
        for i, l in enumerate(self.conv_layers):
            sig = torch.cat([self.conv_state[i], sig], 1)
            sig, skp = l(sig, lc)
            if skp_sum: skp_sum += skp
            else skp_sum = skp
            
        post1 = self.post1(nn.ReLU(skp_sum))
        quant = self.post2(nn.ReLU(post1))
        probs = self.sm(quant) 

        return probs 


