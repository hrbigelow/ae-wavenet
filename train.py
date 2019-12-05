from sys import stderr
from pprint import pprint
import torch
import data
import numpy as np

def main():
    # Initialize data
    dataset = data.Slice(10, 1000)
    dataset.extra_field = torch.ByteTensor(np.random.rand(11338))

    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    device = xm.xla_device()
    wav_loader = data.WavLoader(dataset)
    data_loader = pl.ParallelLoader(wav_loader, [device])
    data_iter = TPULoaderIter(data_loader, device)
    batch_pre = next(data_iter)
    batch = next(data_iter)

if __name__ == '__main__':
    main()

