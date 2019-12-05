from sys import stderr
import torch
import numpy as np

class VirtualBatch(object):
    def __init__(self, dataset):
        super(VirtualBatch, self).__init__()
        self.ds = dataset
        bs = self.ds.batch_size
        self.wav_dec_input = torch.empty((bs, self.ds.window_batch_size))

    def __repr__(self):
        fmt = (
            'wav_dec_input.shape: {}\n'
        )
        return fmt.format(self.wav_dec_input.shape)


class Slice(torch.utils.data.IterableDataset):
    def __init__(self, batch_size, window_batch_size):
        self.init_args = {
                'batch_size': batch_size,
                'window_batch_size': window_batch_size
                }
        self._initialize()


    def _initialize(self):
        super(Slice, self).__init__()
        self.__dict__.update(self.init_args)

    # def __setstate__(self, init_args):
    #     self.init_args = init_args 
    #     self._initialize()

    # def __getstate__(self):
    #     return self.init_args

    def __iter__(self):
        return self

    def __next__(self):
        vb = VirtualBatch(self)
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



def main():
    # Initialize data
    dataset = Slice(10, 1000)
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

