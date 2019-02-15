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

    def forward(x, lc):
        filt = self.conv_signal(x) + self.proj_signal(lc)
        gate = self.conv_gate(x) + self.proj_gate(lc)
        z = torch.tanh(filt) * torch.sigmoid(gate)
        sig = self.dil_res(z)
        skp = self.dil_skp(z)
        sig += x
        return sig, skp 



class WaveNet(nn.Module):
    def __init__(self, n_in, n_kern, n_lc, n_res, n_dil, n_skp, n_post, n_quant,
            n_blocks, n_block_layers):
        self.n_blocks = n_blocks
        self.n_block_layers = n_block_layers

        self.base_layer = nn.Conv1d(n_in, n_res, self.kern_size, self.stride,
                dilation=1, bias=self.bias)
        self.conv_layers = nn.ModuleList() 
        for b in range(self.n_blocks):
            for bl in range(self.n_block_layers):
                dil = bl**2
                self.conv_layers.append(
                        GatedResidualCondConv(n_lc, 2, n_res, n_dil, n_skp, 1, 1, dil))

        self.post1 = nn.Conv1d(n_skp, n_post, 1, 1, 1, bias)
        self.post2 = nn.Conv1d(n_post, n_quant, 1, 1, 1, bias)
        self.softmax = nn.Softmax(1) # Fix this dimension

    def forward(x, lc):
        sig = self.base_layer(x) 
        skp_sum = None
        for l in self.layers:
            sig, skp = l(sig, lc)
            if skp_sum: skp_sum += skp
            else skp_sum = skp
            
        post1 = self.post1(nn.ReLU(skp_sum))
        quant = self.post2(nn.ReLU(post1))
        probs = self.sm(quant) 

        return probs 


