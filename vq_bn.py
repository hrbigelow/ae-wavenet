import torch
from torch import nn
import netmisc
import util


class StopGradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src):
        return src

    @staticmethod
    def backward(ctx, src):
        return src.new_zeros(src.size())

class StopGrad(nn.Module):
    """Implements the StopGradient operation.
    Usage:
    sg = StopGrad()
    a = Tensor(..., requires_grad=True)
    a.grad.zero_()
    b = sg(a).sum()
    b.backward()
    assert (a.grad == 0).all().item()
    """
    def __init__(self):
        super(StopGrad, self).__init__()

    def forward(self, src):
        return StopGradFn.apply(src)


class ReplaceGradFn(torch.autograd.Function):
    """
    This is like a StopGradient operation, except that instead
    of assigning zero gradient to src, assigns gradient of trg to src 
    """
    @staticmethod
    def forward(ctx, src, trg):
        assert src.size() == trg.size()
        return src, trg 

    @staticmethod
    def backward(ctx, src_grad, trg_grad):
        return src_grad.new_zeros(src_grad.size()), src_grad + trg_grad 


class ReplaceGrad(nn.Module):
    """
    Usage:
    rg = ReplaceGrad()
    s1 = Tensor(..., requires_grad=True)
    t1 = Tensor(..., requires_grad=True)
    s2, t2 = rg(s1, t1)

    s1 receives the zero gradient 
    t1 receives the sum of s2's and t2's gradient
    
    """
    def __init__(self):
        super(ReplaceGrad, self).__init__()

    def forward(self, src, trg):
        return ReplaceGradFn.apply(src, trg)

class L2Error(nn.Module):
    def __init__(self):
        super(L2Error, self).__init__()
        self.sg = StopGrad()

    def forward(self, ze, emb):
        """
        B, Q, K, N: n_batch, n_quant_dims, n_quant_vecs, n_timesteps
        ze: (B, Q, N), 
        emb: (K, Q), the quantized embedding vectors
        returns: (B, N) 
        """
        sg_ze = self.sg(ze)
        l2norm_sq = ((sg_ze.unsqueeze(1) - emb.unsqueeze(2)) ** 2).sum(dim=2) # B, K, N
        l2norm_min_val, l2norm_min_ind = l2norm_sq.min(dim=1) # B, N
        l2_error = l2norm_min_val
        return l2_error

class VQ(nn.Module):
    def __init__(self, n_in, n_out, vq_gamma, vq_n_embed):
        super(VQ, self).__init__()
        self.d = n_out
        self.gamma = vq_gamma
        self.k = vq_n_embed 
        self.linear = nn.Conv1d(n_in, self.d, 1, bias=False)
        self.sg = StopGrad()
        self.rg = ReplaceGrad()
        self.ze = None
        self.l2norm_min = None
        self.register_buffer('ind_hist', torch.zeros(self.k))
        self.circ_inds = None
        self.emb = nn.Parameter(data=torch.empty(self.k, self.d))
        netmisc.xavier_init(self.linear)
        nn.init.xavier_uniform_(self.emb, gain=10)

        # Shows how many of the embedding vectors have non-zero gradients
        #self.emb.register_hook(lambda k: print(k.sum(dim=1).unique(sorted=True)))

    def forward(self, z):
        """
        B, Q, K, N: n_batch, n_quant_dims, n_quant_vecs, n_timesteps
        ze: (B, Q, N) 
        emb: (K, Q)
        """
        ze = self.linear(z)

        self.ze = ze
        sg_emb = self.sg(self.emb)
        l2norm_sq = ((ze.unsqueeze(1) - sg_emb.unsqueeze(2)) ** 2).sum(dim=2) # B, K, N
        self.l2norm_min, l2norm_min_ind = l2norm_sq.min(dim=1) # B, N
        zq = util.gather_md(sg_emb, 0, l2norm_min_ind).permute(1, 0, 2)
        zq_rg, __ = self.rg(zq, self.ze)

        # Diagnostics
        ni = l2norm_min_ind.nelement() 
        if self.circ_inds is None:
            self.write_pos = 0
            self.circ_inds = ze.new_full((100, ni), -1, dtype=torch.long)

        self.circ_inds[self.write_pos,0:ni] = l2norm_min_ind.flatten(0)
        self.circ_inds[self.write_pos,ni:] = -1
        self.write_pos += 1
        self.write_pos = self.write_pos % 100

        ones = self.emb.new_ones(ni)
        util.int_hist(l2norm_min_ind, accu=self.ind_hist)
        self.uniq = l2norm_min_ind.unique(sorted=False)
        #self.ze_norm = (self.ze ** 2).sum(dim=1).sqrt()
        #self.emb_norm = (self.emb ** 2).sum(dim=1).sqrt()

        return zq_rg

class VQLoss(nn.Module):
    def __init__(self, bottleneck):
        super(VQLoss, self).__init__()
        self.bn = bottleneck 
        self.logsoftmax = nn.LogSoftmax(1) # input is (B, Q, N)
        self.combine = netmisc.LCCombine('LCCombine')
        self.l2 = L2Error()

    def set_geometry(self, beg_rf, end_rf):
        self.combine.set_geometry(beg_rf, end_rf)

    def forward(self, quant_pred, target_wav):
        # Loss per embedding vector 
        l2_loss_embeds = self.l2(self.bn.ze, self.bn.emb)
        com_loss_embeds = self.bn.l2norm_min * self.bn.gamma
        # l2_loss_embeds = self.l2(self.bn.ze, self.bn.emb).sqrt()
        # com_loss_embeds = self.bn.l2norm_min.sqrt() * self.bn.gamma

        log_pred = self.logsoftmax(quant_pred)
        log_pred_target = torch.gather(log_pred, 1, target_wav.unsqueeze(1))

        # Loss per timestep
        l2_loss_ts = self.combine(l2_loss_embeds.unsqueeze(1))[...,:-1]
        com_loss_ts = self.combine(com_loss_embeds.unsqueeze(1))[...,:-1]
        log_pred_loss_ts = - log_pred_target

        total_loss_ts = log_pred_loss_ts + l2_loss_ts + com_loss_ts
        total_loss = total_loss_ts.mean()

        nh = self.bn.ind_hist / self.bn.ind_hist.sum()

        losses = { 
                'rec': log_pred_loss_ts.mean(),
                'l2': l2_loss_ts.mean(),
                'com': com_loss_ts.mean(),
                'ze_rng': self.bn.ze.max() - self.bn.ze.min(),
                'emb_rng': self.bn.emb.max() - self.bn.emb.min(),
                'hist_ent': util.entropy(self.bn.ind_hist, True),
                'hist_100': util.entropy(util.int_hist(self.bn.circ_inds, -1), True),
                #'p_m': log_pred.max(dim=1)[0].to(torch.float).mean(),
                #'p_sd': log_pred.max(dim=1)[0].to(torch.float).std(),
                'unq': self.bn.uniq,
                #'m_ze': self.bn.ze_norm.max(),
                #'m_emb': self.bn.emb_norm.max()
                }
        netmisc.print_metrics(log_pred, self.bn.emb, losses, 50)

        return total_loss

