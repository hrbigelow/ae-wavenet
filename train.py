from sys import stderr
import torch
import numpy as np
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

class MyBatch(object):
    def __init__(self, dataset):
        super(MyBatch, self).__init__()
        self.ds = dataset
        self.ten = torch.empty((self.ds.dim1, self.ds.dim2))

    def __repr__(self):
        return 'ten.shape: {}'.format(self.ten.shape)


class MyDataset(torch.utils.data.IterableDataset):
    def __init__(self, dim1, dim2):
        self.init_args = { 'dim1': dim1, 'dim2': dim2 }
        self._initialize()

    def _initialize(self):
        super(MyDataset, self).__init__()
        self.__dict__.update(self.init_args)

    def __setstate__(self, init_args):
        self.init_args = init_args 
        self._initialize()

    def __getstate__(self):
        return self.init_args

    def __iter__(self):
        return self

    def __next__(self):
        return MyBatch(self)


class MyLoader(torch.utils.data.DataLoader):
    @staticmethod
    def ident(x):
        return x

    def __init__(self, wav_dataset):
        super(MyLoader, self).__init__(
                dataset=wav_dataset,
                batch_sampler=None,
                collate_fn=self.ident
                )


class TPULoaderIter(object):
    def __init__(self, parallel_loader, device):
        self.per_dev_loader = parallel_loader.per_device_loader(device)

    def __next__(self):
        vb = self.per_dev_loader.__next__()[0]
        return vb


def main():
    dataset = MyDataset(10, 1000)

    # With this line removed, batch1 is not corrupted
    # Or, with this line present, but __getstate__ and __setstate__ removed,
    # batch1 is not corrupted.
    dataset.extra_field = torch.ByteTensor(np.random.rand(11338))

    device = xm.xla_device()
    plain_loader = MyLoader(dataset)
    para_loader = pl.ParallelLoader(plain_loader, [device])
    data_iter = TPULoaderIter(para_loader, device)

    # batch1 is corrupted, batch2 looks good 
    batch1 = next(data_iter)
    batch2 = next(data_iter)
    if batch1.ten.shape == batch2.ten.shape:
        print('Shapes of batch1 and batch2 match')
    else:
        print('Mismatching shapes: batch1: {} != batch2: {}'.format(
            batch1.ten.shape, batch2.ten.shape))

if __name__ == '__main__':
    main()

