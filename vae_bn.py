import torch
from torch import nn
from sys import stderr
import netmisc


class VAE(nn.Module):
    def __init__(self, n_in, n_out, n_sam_per_datapoint=1, bias=True):
        '''n_sam_per_datapoint is L from equation 7,
        https://arxiv.org/pdf/1312.6114.pdf'''
        super(VAE, self).__init__()
        self.linear = nn.Conv1d(n_in, n_out * 2, 1, bias=False)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Conv1d(n_out * 2, n_out * 2, 1, bias=bias)
        self.n_sam_per_datapoint = n_sam_per_datapoint
        _xavier_init(self.linear)
        _xavier_init(self.linear2)

        # Cache these values for later access by the objective function
        self.mu = None
        self.sigma = None

    def forward(self, z):
        # B, T, I, C: n_batch, n_timesteps, n_in_chan, n_out_chan
        # L: n_sam_per_datapoint
        # Input: (B, I, T)
        # Output: (B * L, C, T)
        # lin is the output of 'Linear(128)' from Figure 1 of Chorowski Jan 2019.
        lin = self.linear(z)

        # Chorowski doesn't specify anything between lin and mu/sigma.  But, at
        # the very least, sigma must be positive.  So, I adopt techniques from
        # Appendix C.2, Gaussian MLP as encoder or decoder" from Kingma VAE
        # paper.
        h = self.tanh(lin)
        #h = lin
        mss = self.linear2(h)
        #mss = h
        n_out_chan = mss.size(1) // 2
        mu = mss[:,:n_out_chan,:] 
        log_sigma_sq = mss[:,n_out_chan:,:] 
        sigma_sq = torch.exp(log_sigma_sq)
        sigma = torch.sqrt(sigma_sq)
        # sigma_sq = mss[:,n_out_chan:,:]
        #sigma = torch.sqrt(sigma_sq)

        L = self.n_sam_per_datapoint
        sample_sz = (mu.size()[0] * L,) + mu.size()[1:]
        if L > 1:
            sigma_sq = sigma_sq.repeat(L, 1, 1)
            log_sigma_sq = log_sigma_sq.repeat(L, 1, 1)
            mu = mu.repeat(L, 1, 1)

        # epsilon is the randomness injected here
        epsilon = mu.new_empty(sample_sz).normal_()
        samples = sigma * epsilon + mu 
        # Cache mu and sigma for objective function later 
        self.mu = mu
        self.sigma_sq = sigma_sq
        self.log_sigma_sq = log_sigma_sq
        #print(('linmu: {:.3}, linsd: {:.3}, zmu: {:.3}, zsd: {:.3}, mmu: {:.3}, msd: {:.3}, smu:'
        #        '{:.3}, ssd: {:.3}').format(lin.mean(), lin.std(), z.mean(),
        #            z.std(), mu.mean(), mu.std(), sigma.mean(), sigma.std()))
        return samples

class SGVBLoss(nn.Module):
    def __init__(self, bottleneck):
        super(SGVBLoss, self).__init__()
        self.bottleneck = bottleneck
        self.logsoftmax = nn.LogSoftmax(1) # input is (B, Q, N)
        self.combine = netmisc.LCCombine('LCCombine')

    def set_geometry(self, beg_rf, end_rf):
        self.combine.set_geometry(beg_rf, end_rf)

    def forward(self, quant_pred, target_wav):
        '''
        Compute SGVB estimator from equation 8 in
        https://arxiv.org/pdf/1312.6114.pdf
        Uses formulas from "Autoencoding Variational Bayes",
        Appendix B, "Solution of -D_KL(q_phi(z) || p_theta(z)), Gaussian Case"
        '''
        # B, T, Q, L: n_batch, n_timesteps, n_quant, n_samples_per_datapoint
        # K: n_bottleneck_channels
        # log_pred: (L * B, T, Q), the companded, quantized waveforms.
        # target_wav: (B, T)
        # mu, log_sigma_sq: (B, T, K), the vectors output by the bottleneck
        # Output: scalar, L(theta, phi, x)
        # log_sigma_sq = self.bottleneck.log_sigma_sq
        log_pred = self.logsoftmax(quant_pred)
        sigma_sq = self.bottleneck.sigma_sq
        mu = self.bottleneck.mu
        log_sigma_sq = torch.log(sigma_sq)
        mu_sq = mu * mu

        # neg_kl_div_gaussian: (B, K)
        channel_terms = 1.0 + log_sigma_sq -  mu_sq - sigma_sq 
        chan_dim = 1
        neg_kl_div_gauss = 0.5 * torch.sum(channel_terms, dim=chan_dim, keepdim=True)

        # The last element is a prediction past the end of our target so must trim. 
        combined_kl = self.combine(neg_kl_div_gauss)[...,:-1]

        L = self.bottleneck.n_sam_per_datapoint
        BL = log_pred.size(0)
        assert BL % L == 0 

        target_wav_aug = target_wav.repeat(L, 1).unsqueeze(1)
        log_pred_target = torch.gather(log_pred, 1, target_wav_aug)
        log_pred_target_avg = torch.mean(log_pred_target, dim=1, keepdim=True)

        sgvb = - combined_kl - log_pred_target_avg  
        total_loss = sgvb.mean()

        log_pred_loss = - log_pred_target_avg.mean()
        kl_div_loss = - combined_kl.mean()

        peak_mean = log_pred.max(dim=1)[0].to(torch.float).mean()
        peak_sd = log_pred.max(dim=1)[0].to(torch.float).std()

        fmt='kl_div_loss: {:.5f}, log_pred_loss: {:.5f}, peak_mean: {:.3f}, peak_sd: {:.3f}'
        print(fmt.format(kl_div_loss, log_pred_loss, peak_mean, peak_sd), file=stderr)
        stderr.flush()

        return total_loss 

