import torch
from torch import nn
import netmisc
import util


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
        self.min_dist = None
        self.register_buffer('ind_hist', torch.zeros(self.k))
        self.circ_inds = None
        self.emb = nn.Parameter(data=torch.empty(self.k, self.d))
        nn.init.xavier_uniform_(self.emb, gain=1)

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
        self.min_dist, min_ind = l2norm_sq.min(dim=1) # B, N
        zq = util.gather_md(sg_emb, 0, min_ind).permute(1, 0, 2)
        zq_rg, __ = self.rg(zq, self.ze)

        # Diagnostics
        ni = min_ind.nelement() 
        if self.circ_inds is None:
            self.write_pos = 0
            self.circ_inds = ze.new_full((100, ni), -1, dtype=torch.long)

        self.circ_inds[self.write_pos,0:ni] = min_ind.flatten(0)
        self.circ_inds[self.write_pos,ni:] = -1
        self.write_pos += 1
        self.write_pos = self.write_pos % 100

        ones = self.emb.new_ones(ni)
        util.int_hist(min_ind, accu=self.ind_hist)
        self.uniq = min_ind.unique(sorted=False)
        self.ze_norm = (self.ze ** 2).sum(dim=1).sqrt()
        self.emb_norm = (self.emb ** 2).sum(dim=1).sqrt()

        return zq_rg

class VQLoss(nn.Module):
    def __init__(self, bottleneck):
        super(VQLoss, self).__init__()
        self.bn = bottleneck 
        self.logsoftmax = nn.LogSoftmax(1) # input is (B, Q, N)
        # self.combine = netmisc.LCCombine('LCCombine')
        # self.usage_adjust = netmisc.EmbedLossAdjust('EmbedLossAdjust')
        self.l2 = L2Error()

    def forward(self, quant_pred, target_wav):
        """
        quant_pred: 
        target_wav: B,  
        """
        # Loss per embedding vector 
        l2_loss_embeds = self.l2(self.bn.sg(self.bn.ze), self.bn.emb)
        # l2_loss_embeds = scaled_l2_norm(self.bn.sg(self.bn.ze), self.bn.emb)
        com_loss_embeds = self.bn.min_dist * self.bn.gamma

        log_pred = self.logsoftmax(quant_pred)
        log_pred_target = torch.gather(log_pred, 1,
                target_wav.long().unsqueeze(1))

        # Loss per timestep
        # !!! We don't need a 'loss per timestep'.  We only need
        # to adjust the l2 and com losses by usage weight of each
        # code.  (The codes at the two ends of the window will be
        # used less)
        rec_loss_ts = - log_pred_target

        # Use only a subset of the overlapping windows
        #sl = slice(0, 1)
        #rec_loss_sel = rec_loss_ts[...,sl]
        #l2_loss_sel = l2_loss_ts[...,sl]
        #com_loss_sel = com_loss_ts[...,sl]
        
        # total_loss_sel = rec_loss_sel + l2_loss_sel + com_loss_sel
        # total_loss_ts = l2_loss_ts
        # total_loss_ts = com_loss_ts
        # total_loss_ts = com_loss_ts + l2_loss_ts
        # total_loss_ts = log_pred_loss_ts + l2_loss_ts
        # total_loss_ts = log_pred_loss_ts 
        # total_loss_ts = com_loss_ts - com_loss_ts

        # total_loss = total_loss_sel.mean()

        # We use sum here for each of the three loss terms because each element
        # should affect the total loss equally.  For a typical WaveNet
        # architecture, there will be only one l2 loss term (or com_loss term)
        # per 320 rec_loss terms, due to upsampling.  We could adjust for that.
        # Implicitly, com_loss is already adjusted by gamma.  Perhaps l2_loss
        # should also be adjusted, but at the moment it is not.
        total_loss = rec_loss_ts.sum() + l2_loss_embeds.sum() + com_loss_embeds.sum()

        nh = self.bn.ind_hist / self.bn.ind_hist.sum()

        self.metrics = { 
                'rec': rec_loss_ts.mean(),
                'l2': l2_loss_embeds.mean(),
                'com': com_loss_embeds.mean(),
                #'ze_rng': self.bn.ze.max() - self.bn.ze.min(),
                #'emb_rng': self.bn.emb.max() - self.bn.emb.min(),
                'min_ze': self.bn.ze_norm.min(),
                'max_ze': self.bn.ze_norm.max(),
                'min_emb': self.bn.emb_norm.min(),
                'max_emb': self.bn.emb_norm.max(),
                'hst_ent': util.entropy(self.bn.ind_hist, True),
                'hst_100': util.entropy(util.int_hist(self.bn.circ_inds, -1), True),
                #'p_m': log_pred.max(dim=1)[0].to(torch.float).mean(),
                #'p_sd': log_pred.max(dim=1)[0].to(torch.float).std(),
                'nunq': self.bn.uniq.nelement(),
                'pk_m': log_pred.max(dim=1)[0].to(torch.float).mean(),
                'pk_nuq': log_pred.max(dim=1)[1].unique().nelement(),
                # 'peak_unq': log_pred.max(dim=1)[1].unique(),
                'pk_sd': log_pred.max(dim=1)[0].to(torch.float).std(),
                # 'unq': self.bn.uniq,
                #'m_ze': self.bn.ze_norm.max(),
                #'m_emb': self.bn.emb_norm.max()
                #emb0 = emb - emb.mean(dim=0)
                #chan_var = (emb0 ** 2).sum(dim=0)
                #chan_covar = torch.matmul(emb0.transpose(1, 0), emb0) - torch.diag(chan_var)
                }
        # netmisc.print_metrics(losses, 10000000)

        return total_loss

