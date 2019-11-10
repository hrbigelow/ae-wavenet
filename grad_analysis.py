import torch

# Functions for analyzing the gradients during training

# Approach: At a given moment during training, calculate the gradients on each
# weight for N minibatches of data.  Then, calculate the standard deviation on
# each weight over the N minibatches.  Finally, report the average standard
# deviation, and perhaps quantiles, over the weights in a given layer

# This will inform whether the batch size is too small and thus too noisy
# for a given learning rate

# We need to have a function for copying the gradients after a call to
# backward, into some larger vector with an extra dimension.

# Due to memory constraints, we cannot store all N sets of gradients for all
# parameters, nor is it efficient to store one at a time and re-run the
# forward/backward pass for each parameter set.  Instead, we use an incremental
# formula for the variance, from
# http://datagenetics.com/blog/november22017/index.html:

# mu_0 = x_0, S_0 = 0
# mu_n = mu_(n-1) + (x_n - mu_(n-1)) / n
# S_n = S_(n-1) + (x_n - mu_(n-1)) (x_n - mu_n)
# sigma_n = sqrt(S_n / n)
def mu_s_incr(x_cur, n, mu_pre, s_pre):
    """
    Calculate current mu and s from previous values using incremental formula.
    All three arguments are assumed to have the same shape and are computed
    elementwise
    """
    if n == 0:
        return x_cur, x_cur.new_zeros(x_cur.shape)

    assert x_cur.shape == mu_pre.shape
    assert x_cur.shape == s_pre.shape
    mu_cur = mu_pre + (x_cur - mu_pre) / n
    s_cur = s_pre + (x_cur - mu_pre) * (x_cur - mu_cur)
    return mu_cur, s_cur


def quantiles(x, quantiles):
    """
    Return the quantiles of x.  quantiles are given in [0, 1]
    """
    qv = [0] * len(quantiles)
    for i, q in enumerate(quantiles):
        k = 1 + round(float(q) * (x.numel() - 1))
        qv[i] = x.view(-1).kthvalue(k)[0].item()
    return qv


def grad_stats(model, update_model_closure, n_batch, report_quantiles):
    """
    Run n_batch'es of data through the model, accumulating an incremental
    mean and sd of the gradients.  Report the quantiles of these sigma values
    per parameter.
    model is a torch.Module
    update_model_closure should fetch a new batch of data, then run
    forward()/backward() to update the gradients
    """
    mu = {}
    s = {}

    update_model_closure()
    
    for name, par in model.named_parameters():
        if par.grad is None:
            continue
        mu[name] = None 
        s[name] = None 

    for b in range(n_batch):
        update_model_closure()
        for name, par in model.named_parameters():
            if par.grad is None:
                continue
            mu[name], s[name] = mu_s_incr(par.grad, b, mu[name], s[name])

    quantile_values = {}
    for name, sval in s.items():
        sig = (sval / n_batch).sqrt().cpu()
        quantile_values[name] = quantiles(sig, report_quantiles)

    return quantile_values

