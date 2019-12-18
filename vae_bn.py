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
        # self.linear_mu = nn.Conv1d(n_out, n_out, 1, bias=bias)
        # self.linear_sigma = nn.Conv1d(n_out, n_out, 1, bias=bias)
        self.n_sam_per_datapoint = n_sam_per_datapoint
        self.n_out_chan = n_out
        netmisc.xavier_init(self.linear)
        # netmisc.xavier_init(self.linear_mu)
        # netmisc.xavier_init(self.linear_sigma)

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
        mu, log_sigma_sq = torch.split(lin, self.n_out_chan, dim=1)
        sigma = torch.exp(0.5 * log_sigma_sq)
        # sigma_sq = mss[:,n_out_chan:,:]
        #sigma = torch.sqrt(sigma_sq)

        L = self.n_sam_per_datapoint
        sample_sz = (mu.size()[0] * L,) + mu.size()[1:]
        if L > 1:
            sigma_sq = sigma_sq.repeat(L, 1, 1)
            log_sigma_sq = log_sigma_sq.repeat(L, 1, 1)
            mu = mu.repeat(L, 1, 1)

        # epsilon is the randomness injected here
        samples = torch.randn_like(mu)
        samples.mul_(sigma)
        samples.add_(mu)

        # Cache mu and sigma for objective function later 
        self.mu = mu
        self.sigma_sq = torch.pow(sigma, 2.0)
        self.log_sigma_sq = log_sigma_sq
        #print(('linmu: {:.3}, linsd: {:.3}, zmu: {:.3}, zsd: {:.3}, mmu: {:.3}, msd: {:.3}, smu:'
        #        '{:.3}, ssd: {:.3}').format(lin.mean(), lin.std(), z.mean(),
        #            z.std(), mu.mean(), mu.std(), sigma.mean(), sigma.std()))
        return samples

class SGVBLoss(nn.Module):
    def __init__(self, bottleneck, free_nats):
        super(SGVBLoss, self).__init__()
        self.bottleneck = bottleneck
        self.register_buffer('free_nats', torch.tensor(free_nats))
        self.register_buffer('anneal_weight', torch.tensor(0.0))
        self.logsoftmax = nn.LogSoftmax(1) # input is (B, Q, N)

    def update_anneal_weight(self, anneal_weight):
        self.anneal_weight.fill_(anneal_weight)
        

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

        # neg_kl_div_gaussian: (B, K) (from Appendix B at end of derivation)
        channel_terms = 1.0 + log_sigma_sq -  mu_sq - sigma_sq 
        neg_kl_div_gauss = 0.5 * torch.sum(channel_terms)

        L = self.bottleneck.n_sam_per_datapoint
        BL = log_pred.size(0)
        assert BL % L == 0 

        target_wav_aug = target_wav.repeat(L, 1).unsqueeze(1).long()
        log_pred_target = torch.gather(log_pred, 1, target_wav_aug)
        log_pred_target_avg = torch.mean(log_pred_target)

        log_pred_loss = - log_pred_target_avg
        kl_div_loss = - neg_kl_div_gauss 

        # "For the VAE, this collapse can be prevented by annealing the weight
        # of the KL term and using the free-information formulation in Eq. (2)"
        # (See p 3 Section C second paragraph)
        total_loss = (
            log_pred_loss + self.anneal_weight 
            * torch.clamp(kl_div_loss, min=self.free_nats))

        self.metrics = {
                'kl_div_loss': kl_div_loss,
                'log_pred_loss': log_pred_loss,
                # 'mu_abs_max': mu.abs().max(),
                # 's_sq_abs_max': sigma_sq.abs().max()
                }

        return total_loss 

