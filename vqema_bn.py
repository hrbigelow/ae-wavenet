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


def scaled_l2_norm(z, q):
    """
    Computes a distance D(z, q) with properties:
    D(lambda * z, lambda * q) = D(z, q)
    D(z, 0) = D(0, z) = 1
    D(z, lambda*z) = |1-lambda| / (1 + |lambda|)
    """
    num = ((z - q) ** 2).sum(dim=2).sqrt()
    den = (z ** 2).sum(dim=2).sqrt() + (q ** 2).sum(dim=2).sqrt()
    return num / den
    

class VQEMA(nn.Module):
    """
    Vector Quantization bottleneck using Exponential Moving Average
    updates of the Codebook vectors.
    """
    def __init__(self, n_in, n_out, vq_gamma, vq_ema_gamma, vq_n_embed, training):
        super(VQEMA, self).__init__()
        self.training = training
        self.d = n_out
        self.gamma = vq_gamma
        self.ema_gamma = vq_ema_gamma
        self.ema_gamma_comp = 1.0 - self.ema_gamma
        self.k = vq_n_embed 
        self.linear = nn.Conv1d(n_in, self.d, 1, bias=False)
        self.sg = StopGrad()
        self.rg = ReplaceGrad()
        self.ze = None
        self.register_buffer('emb', torch.empty(self.k, self.d))
        nn.init.xavier_uniform_(self.emb, gain=10)

        if self.ema_gamma >= 1.0 or self.ema_gamma <= 0:
            raise RuntimeError('VQEMA must use an EMA-gamma value in (0, 1)')

        if self.training:
            self.min_dist = None
            self.circ_inds = None
            self.register_buffer('ind_hist', torch.zeros(self.k))
            self.register_buffer('ema_numer', torch.empty(self.k, self.d))
            self.register_buffer('ema_denom', torch.empty(self.k))
            self.register_buffer('z_sum', torch.empty(self.k, self.d))
            self.register_buffer('n_sum', torch.empty(self.k))
            self.register_buffer('n_sum_ones', torch.ones(self.k))
            #self.ema_numer.detach_()
            #self.ema_denom.detach_()
            #self.z_sum.detach_()
            #self.n_sum.detach_()
            #self.emb.detach_()
            #nn.init.ones_(self.ema_denom)
            self.ema_numer = self.emb * self.ema_gamma_comp
            self.ema_denom = self.n_sum_ones * self.ema_gamma_comp

        netmisc.xavier_init(self.linear)

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
        # self.min_dist, min_ind = l2norm_sq.min(dim=1) # B, N

        snorm = scaled_l2_norm(ze.unsqueeze(1),
                sg_emb.unsqueeze(2).unsqueeze(0))
        #print('snorm: ', snorm)
        self.min_dist, min_ind = snorm.min(dim=1) # B, N
        zq = util.gather_md(sg_emb, 0, min_ind).permute(1, 0, 2)

        if self.training:
            # Diagnostics
            ni = min_ind.nelement() 
            #if self.circ_inds is None:
            #    self.write_pos = 0
            #    self.circ_inds = ze.new_full((100, ni), -1, dtype=torch.long)

            #self.circ_inds[self.write_pos,0:ni] = min_ind.flatten(0)
            #self.circ_inds[self.write_pos,ni:] = -1
            #self.write_pos += 1
            #self.write_pos = self.write_pos % 100
            ones = self.emb.new_ones(ni)
            util.int_hist(min_ind, accu=self.ind_hist)
            self.uniq = min_ind.unique(sorted=False)
            self.ze_norm = (self.ze ** 2).sum(dim=1).sqrt()
            self.emb_norm = (self.emb ** 2).sum(dim=1).sqrt()
            self.min_ind = min_ind

            # EMA statistics
            # min_ind: B, W
            # ze: B, D, W
            # z_sum: K, D
            # n_sum: K
            # scatter_add has the limitation that the size of the indexing
            # vector cannot exceed that of the destination (even in the target
            # indexing dimension, which doesn't make much sense)
            # In this case, K is the indexing dimension
            # batch_size * window_batch_size
            flat_ind = min_ind.flatten(0, 1)
            idim = max(flat_ind.shape[0], self.k)

            z_tmp_shape = [idim, self.d]
            n_sum_tmp = self.n_sum.new_zeros(idim)

            z_sum_tmp = self.z_sum.new_zeros(z_tmp_shape)
            z_sum_tmp.scatter_add_(0,
                    flat_ind.unsqueeze(1).repeat(1, self.d),
                    self.ze.permute(0,2,1).flatten(0, 1)
                    )
            self.z_sum[...] = z_sum_tmp[0:self.k,:]

            self.n_sum.zero_()
            n_sum_ones = n_sum_tmp.new_ones((idim))
            n_sum_tmp.scatter_add_(0, flat_ind, n_sum_ones)
            self.n_sum[...] = n_sum_tmp[0:self.k]

            self.ema_numer = (
                    self.ema_gamma * self.ema_numer +
                    self.ema_gamma_comp * self.z_sum) 
            self.ema_denom = (
                    self.ema_gamma * self.ema_denom +
                    self.ema_gamma_comp * self.n_sum)

            # construct the straight-through estimator ('ReplaceGrad')
            # What I need is 
            # cb_update = self.ema_numer / self.ema_denom.unsqueeze(1).repeat(1,
            #         self.d)

            # print('z_sum_norm:', (self.z_sum ** 2).sum(dim=1).sqrt())
            # print('n_sum_norm:', self.n_sum)
            print('ze_norm:', self.ze_norm)
            print('emb_norm:', (self.emb ** 2).sum(dim=1).sqrt())
            print('min_ind:', self.min_ind)
            # print('cb_update_norm:', (cb_update ** 2).sum(dim=1).sqrt())
            # print('ema_numer_norm:',
            #         (self.ema_numer ** 2).sum(dim=1).sqrt().mean())
            # print('ema_denom_norm:',
            #         (self.ema_denom ** 2).sqrt().mean())
            zq_rg, __ = self.rg(zq, self.ze)

        return zq_rg

    def update_codebook(self):
        """
        Updates the codebook based on the EMA statistics
        """
        self.emb = self.ema_numer / self.ema_denom.unsqueeze(1).repeat(1,
                self.d)
        self.emb.detach_()


class VQEMALoss(nn.Module):
    def __init__(self, bottleneck):
        super(VQEMALoss, self).__init__()
        self.bn = bottleneck 
        self.logsoftmax = nn.LogSoftmax(1) # input is (B, Q, N)

    def forward(self, quant_pred, target_wav):
        """
        quant_pred: 
        target_wav: B,  
        """
        # Loss per embedding vector 
        com_loss_embeds = self.bn.min_dist * self.bn.gamma

        log_pred = self.logsoftmax(quant_pred)
        log_pred_target = torch.gather(log_pred, 1,
                target_wav.long().unsqueeze(1))

        rec_loss_ts = - log_pred_target
        # total_loss = rec_loss_ts.sum() + com_loss_embeds.sum()
        # total_loss = rec_loss_ts.sum()
        total_loss = com_loss_embeds.sum()
        # total_loss = com_loss_embeds.sum() * 0.0 

        nh = self.bn.ind_hist / self.bn.ind_hist.sum()

        self.metrics = { 
                'rec': rec_loss_ts.mean(),
                'com': com_loss_embeds.mean(),
                'min_ze': self.bn.ze_norm.min(),
                'max_ze': self.bn.ze_norm.max(),
                'min_emb': self.bn.emb_norm.min(),
                'max_emb': self.bn.emb_norm.max(),
                'hst_ent': util.entropy(self.bn.ind_hist, True),
                # 'hst_100': util.entropy(util.int_hist(self.bn.circ_inds, -1), True),
                'nunq': self.bn.uniq.nelement(),
                'pk_m': log_pred.max(dim=1)[0].to(torch.float).mean(),
                'pk_nuq': log_pred.max(dim=1)[1].unique().nelement(),
                'pk_sd': log_pred.max(dim=1)[0].to(torch.float).std()
                }

        return total_loss

