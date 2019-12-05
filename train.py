from sys import stderr
import torch
import numpy as np
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

class VirtualBatch(object):
    def __init__(self, dim1, dim2):
        super(VirtualBatch, self).__init__()
        self.ten = torch.empty(dim1, dim2)

    def __repr__(self):
        return 'ten.shape: {}'.format(self.ten.shape)


class Slice(torch.utils.data.IterableDataset):
    def __init__(self, dim1, dim2):
        self.init_args = { 'dim1': dim1, 'dim2': dim2 }
        self._initialize()


    def _initialize(self):
        super(Slice, self).__init__()
        self.__dict__.update(self.init_args)

    def __setstate__(self, init_args):
        self.init_args = init_args 
        self._initialize()

    def __getstate__(self):
        return self.init_args

    def __iter__(self):
        return self

    def __next__(self):
        vb = VirtualBatch(self.dim1, self.dim2)
        return vb 


class WavLoader(torch.utils.data.DataLoader):
    """
    Data loader which may be wrapped by a
    torch_xla.distributed.parallel_loader.
    This loader returns batches of tensors on cpu, optionally
    pushing them to target_device if provided
    """
    @staticmethod
    def ident(x):
        return x

    def __init__(self, wav_dataset, target_device=None):
        self.target_device = target_device
        super(WavLoader, self).__init__(
                dataset=wav_dataset,
                batch_sampler=None,
                collate_fn=self.ident
                )

    def set_target_device(self, target_device):
        self.dataset.set_target_device(target_device)


class TPULoaderIter(object):
    def __init__(self, parallel_loader, device):
        self.per_dev_loader = parallel_loader.per_device_loader(device)

    def __next__(self):
        vb = self.per_dev_loader.__next__()[0]
        return vb


def main():
    # Initialize data
    dataset = Slice(10, 1000)
    dataset.extra_field = torch.ByteTensor(np.random.rand(11338))

    device = xm.xla_device()
    wav_loader = WavLoader(dataset)
    data_loader = pl.ParallelLoader(wav_loader, [device])
    data_iter = TPULoaderIter(data_loader, device)
    batch_pre = next(data_iter)
    batch = next(data_iter)

if __name__ == '__main__':
    main()

