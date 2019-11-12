import torch
from torch import nn
import netmisc

class AE(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(AE, self).__init__()
        self.linear = nn.Conv1d(n_in, n_out, 1, bias=bias)
        netmisc.xavier_init(self.linear)

    def forward(self, x):
        """
        ze: (B, Q, N)
        """
        self.ze = self.linear(x)
        return self.ze 


class AELoss(nn.Module):
    def __init__(self, bottleneck, norm_gamma):
        super(AELoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(1) # input is (B, Q, N)
        self.bottleneck = bottleneck
        self.norm_gamma = norm_gamma

    def forward(self, quant_pred, target_wav):

        log_pred = self.logsoftmax(quant_pred)
        target_wav_gather = target_wav.long().unsqueeze(1)
        log_pred_target = torch.gather(log_pred, 1, target_wav_gather)

        rec_loss = - log_pred_target.mean()
        ze_norm = (self.bottleneck.ze ** 2).sum(dim=1).sqrt()

        norm_loss = self.norm_gamma * torch.abs(ze_norm - 1.0).mean()
        total_loss = rec_loss + norm_loss

        self.metrics = {
                'rec': rec_loss,
                'norm': norm_loss
                }

        return total_loss 

