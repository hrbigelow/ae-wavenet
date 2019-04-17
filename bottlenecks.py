from torch import nn
from torch import distributions as dist

class VQVAE(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(VQVAE, self).__init__()
        self.proto_vec = 0
        self.linear = nn.Conv1d(n_in, n_out, 1, bias=bias)
        self.nearest_proto()

    def forward(self, x):
        out = self.linear(x)
        out = self.nearest_proto(out)
        return out


class VAE(nn.Module):
    def __init__(self, n_in, n_out, n_sam_per_datapoint=1, bias=True):
        '''n_sam_per_datapoint is L from equation 7,
        https://arxiv.org/pdf/1312.6114.pdf'''
        super(VAE, self).__init__()
        self.linear = nn.Conv1d(n_in, n_out * 2, 1, bias=bias)
        self.n_sam_per_datapoint = n_sam_per_datapoint

    def forward(self, x):
        # B, T, I, C: n_batch, n_timesteps, n_in_chan, n_out_chan
        # L: n_sam_per_datapoint
        # Input: (B, I, T)
        # Output: (B * L, C, T)
        mu_sigma = self.linear(x)
        n_out_chan = mu_sigma.size(1) // 2
        mu = mu_sigma[:,:n_out_chan,:]  # first half of the channels
        sigma = mu_sigma[:,n_out_chan:,:] # second half of the channels

        # Randomness is injected here
        L = self.n_sam_per_datapoint
        sz = mu.size()
        if L == 1:
            epsilon = mu.new_empty().normal_()
            samples = sigma * epsilon + mu 
        else:
            sz[0] *= L
            epsilon = mu.new_empty(sz).normal_()
            sigma = sigma.repeat(L, 1, 1)
            mu = mu.repeat(L, 1, 1)
            samples = sigma * epsilon + mu
        return samples, sigma, mu 

class SGVB(nn.Module):
    def __init__(self, bottleneck):
        super(SGVB, self).__init__()
        self.bottleneck = bottleneck

    def forward(self, log_pred, mu, sigma, target_wav):
        '''
        Compute SGVB estimator from equation 8 in
        https://arxiv.org/pdf/1312.6114.pdf
        Uses formulas from "Autoencoding Variational Bayes",
        Appendix B, "Solution of -D_KL(q_phi(z) || p_theta(z)), Gaussian Case"
        '''
        # B, T, Q, L: n_batch, n_timesteps, n_quant, n_samples_per_datapoint
        # K: n_bottleneck_channels
        # log_pred: (L * B, T, Q), the companded, quantized waveforms.
        # mu, sigma: (B, T, K), the vectors output by the bottleneck
        # target_wav: (B, T)
        # Output: scalar
        sigma_sq = sigma * sigma
        mu_sq = mu * mu

        # neg_kl_div_gaussian: (B, T)
        neg_kl_div_gaussian = 0.5 * torch.sum(1.0 + torch.log(sigma_sq) - mu_sq -
                sigma_sq, dim=2)

        L = self.bottleneck.n_sam_per_datapoint
        BL = log_pred.size(0)
        assert BL % L == 0 

        tsz = target_wav.size()
        target_wav_aug = target_wav.repeat(L, 1)
        target_probs = torch.gather(log_pred, target_wav_aug, dim=2)
        target_probs_avg = torch.mean(target_probs, dim=1)
        return torch.mean(neg_kl_div_gaussian + target_probs)


class AE(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(AE, self).__init__()
        self.linear = nn.Conv1d(n_in, n_out, 1, bias=bias)

    def forward(self, x):
        out = self.linear(x)
        return out

