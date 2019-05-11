import torch
from torch import nn


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
        emb: (K, N), the quantized embedding vectors
        """
        sg_ze = self.sg(ze)
        l2n_sq = ((sg_ze.unsqueeze(2) - emb) ** 2).sum(dim=1) # B, K, N
        l2n_min_val, __ = l2n_sq.min(dim=1) # B, N
        l2_error = l2n_min_val
        return l2_error

class VQ(nn.Module):
    def __init__(self, n_latent_dim, n_embed_vecs):
        super(VQ, self).__init__()
        self.sg = StopGrad()
        self.rg = ReplaceGrad()
        self.d = n_latent_dim 
        self.k = n_embed_vecs
        self.ze = None
        self.l2norm_min = None
        self.register_buffer('emb', torch.empty((self.k, self.d), requires_grad=True))

    def forward(self, ze):
        """
        B, Q, K, N: n_batch, n_quant_dims, n_quant_vecs, n_timesteps
        ze: (B, Q, N), 
        """
        self.ze = ze
        sg_emb = self.sg(self.emb)
        l2n_sq = ((ze.unsqueeze(2) - sg_emb) ** 2).sum(dim=1) # B, K, N
        self.l2norm_min, l2norm_min_ind = l2n_sq.min(dim=1) # B, N
        zq = torch.index_select(self.ze, 0, l2norm_min_ind) # B, Q, N
        zq_rg, __ = self.rg(zq, self.ze)
        return zq_rg

class VQLoss(nn.Module):
    def __init__(self, bottleneck, beta):
        super(VQLoss, self).__init__()
        self.bottleneck = bottleneck 
        self.beta = beta
        self.logsoftmax = nn.LogSoftmax(1) # input is (B, Q, N)
        self.combine = LCCombine('LCCombine')
        self.l2 = L2Error()

    def forward(self, quant_pred, target_wav):
        l2_loss = self.l2(self.bottleneck.ze, self.bottleneck.emb)
        com_loss = self.l2norm_min * self.beta

        l2_loss_comb = self.combine(l2_loss)[...,:-1]
        com_loss_comb = self.combine(com_loss)[...,:-1]

        log_pred = self.logsoftmax(quant_pred)
        log_pred_target = torch.gather(log_pred, 1, target_wav)

        total_loss_terms = log_pred_target + l2_loss_comb + com_loss_comb
        total_loss = total_loss_terms.mean()

        return total_loss

