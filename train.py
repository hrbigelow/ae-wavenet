import sys
from sys import stderr
from pprint import pprint
import torch

import data

def main():
    # Initialize data
    dataset = data.Slice(10, 1000)
    
    dataset.extra_field = torch.ByteTensor(np.random.rand(11338))

    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    self.device = xm.xla_device()
    wav_loader = data.WavLoader(dataset)
    self.data_loader = pl.ParallelLoader(wav_loader, [self.device])
    self.data_iter = TPULoaderIter(self.data_loader, self.device)
    batch_pre = next(self.data_iter)
    batch = next(self.data_iter)

if __name__ == '__main__':
    main()

