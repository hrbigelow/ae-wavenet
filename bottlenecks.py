import torch
from torch import nn
from torch import distributions as dist
import rfield

class LCCombine(nn.Module):
    '''The bottleneck loss terms for VAE and VQVAE are based on "z", which, for
    a single sample, consists of ~14 or so individual timesteps.  This enables
    the proper accounting of commitment loss terms when batching training
    samples across the time dimension.  
    '''
    def __init__(self, name=None):
        super(LCCombine, self).__init__()
        self.name = name

    def set_geometry(self, beg_rf, end_rf):
        '''
        Constructs the transpose convolution which mimics the usage pattern
        of WaveNet's local conditioning vectors and output.
        '''
        self.rf = rfield.condensed(beg_rf, end_rf, self.name) 
        self.rf.gen_stats(1, self.rf)
        stride = self.rf.stride_ratio.denominator
        l_off, r_off = rfield.offsets(self.rf, self.rf)
        filter_sz = l_off - r_off + 1
        # pad_add = kernel_size - 1 - pad_arg (see torch.nn.ConvTranspose1d)
        # => pad_arg = kernel_size - 1 - pad_add 
        pad_add = max(self.rf.l_pad, self.rf.r_pad)
        self.l_trim = pad_add - self.rf.l_pad
        self.r_trim = pad_add - self.rf.r_pad
        pad_arg = filter_sz - 1 - pad_add
        self.tconv = nn.ConvTranspose1d(1, 1, filter_sz, stride, pad_arg, bias=False)
        self.tconv.weight.requires_grad = False

    def forward(self, z_metric):
        '''
        B, T, S: batch_sz, timesteps, less-frequent timesteps
        D: number of 
        z_metric: B, S, 1 
        output: B, T, 1
        '''
        out = self.tconv(z_metric)
        out_trim = out[:,self.l_trim:-self.r_trim or None,:]
        return out_trim


class VQVAE(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(VQVAE, self).__init__()
        self.proto_vec = 0
        self.linear = nn.Conv1d(n_in, n_out, 1, bias=bias)
        self.nearest_proto()

    def forward(self, x):
        '''
        vq_norms[:,:,0] = ||sg(z_{e}(x)) - e_{q(x)}||^2
        vq_norms[:,:,1] = ||z_{e}(x) - sg(e_{q(x)})||^2

        output[b,t,0] is the sum of the first vq_norm corresponding to the
        conditioning vectors used by WaveNet's output timestep t.  It is used
        as the total commitment loss for sample t
        '''
        out = self.linear(x)
        out = self.nearest_proto(out)
        return out


class VAE(nn.Module):
    def __init__(self, n_in, n_out, n_sam_per_datapoint=1, bias=True):
        '''n_sam_per_datapoint is L from equation 7,
        https://arxiv.org/pdf/1312.6114.pdf'''
        super(VAE, self).__init__()
        self.linear = nn.Conv1d(n_in, n_out * 2, 1, bias=False)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Conv1d(n_out * 2, n_out * 2, 1, bias=bias)
        self.n_sam_per_datapoint = n_sam_per_datapoint

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
        self.combine = LCCombine('LCCombine')

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
        channel_dim = 1
        channel_terms = 1.0 + log_sigma_sq - mu_sq - sigma_sq 
        neg_kl_div_gaussian = \
                0.5 * torch.sum(channel_terms, dim=channel_dim, keepdim=True)
        combined_kl = self.combine(neg_kl_div_gaussian)

        L = self.bottleneck.n_sam_per_datapoint
        BL = log_pred.size(0)
        assert BL % L == 0 

        # Note that, unlike torch.nn.CrossEntropy loss, the second term in
        # equation 7 is a function only of the probabilities assigned to the
        # target quantum, if I'm interpreting the formula correctly.  For
        # example, if the output probabilities are out_prob = [0.1, 0.5, 0.1,
        # 0.3] and the target = 1, then only the out_prob[target] value affects
        # the SGVB second term, not out_prob[i != target].
        
        target_wav_aug = target_wav.repeat(L, 1).unsqueeze(1)
        log_pred_target = torch.gather(log_pred, 1, target_wav_aug)
        log_pred_target_avg = torch.mean(log_pred_target, dim=1, keepdim=True)

        # The last element is a prediction past the end of our target so must trim. 
        sgvb = combined_kl[...,:-1] + log_pred_target_avg  
        #sgvb = combined_kl[...,:-1]
        #sgvb = - neg_kl_div_gaussian
        #sgvb = mu_sq 
        kl_div_loss = - combined_kl.mean()
        log_pred_loss = - log_pred_target_avg.mean()
        peak_mean = log_pred.max(dim=1)[0].to(torch.float).mean()
        peak_sd = log_pred.max(dim=1)[0].to(torch.float).std()
        print(('kl_div_loss: {:.5f}, log_pred_loss: {:.5f}, peak_mean: {:.3f}, '
                'peak_sd: {:.3f}').format(kl_div_loss, log_pred_loss, peak_mean, peak_sd))
        minibatch_sgvb = torch.mean(sgvb)
        # sgvb is to be maximized.  we return the negative and minimize it
        return - minibatch_sgvb


class AE(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(AE, self).__init__()
        self.linear = nn.Conv1d(n_in, n_out, 1, bias=bias)

    def forward(self, x):
        out = self.linear(x)
        return out

