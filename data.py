# Preprocess Data
from sys import stderr, exit
import pickle
import numpy as np
import torch
import torch.utils.data
import jitter
from torch import nn
import vconv
import copy
from collections import namedtuple

import util
import mfcc


def parse_catalog(sam_file):
    try:
        catalog = []
        with open(sam_file) as sam_fh:
            for s in sam_fh.readlines():
                (vid, wav_path) = s.strip().split('\t')
                catalog.append([int(vid), wav_path])
    except (FileNotFoundError, IOError):
        raise RuntimeError("Couldn't open or read samples file {}".format(sam_file))
    return catalog

def convert(catalog, dat_file, n_quant, sample_rate=16000):
    """
    Convert all input data and save a dat file 
    """
    import librosa
    if n_quant <= 2**8:
        snd_dtype = np.uint8
    elif n_quant <= 2**15:
        snd_dtype = np.int16
    else:
        snd_dtype = np.int32

    n_mel_chan = None
    speaker_ids = set(id for id,__ in catalog)
    speaker_id_map = dict((v,k) for k,v in enumerate(speaker_ids))
    snd_data = np.empty((0), dtype=snd_dtype) 
    samples = []

    for (voice_id, snd_path) in catalog:
        snd, _ = librosa.load(snd_path, sample_rate)
        snd_mu = util.mu_encode_np(snd, n_quant).astype(snd_dtype)
        wav_b = len(snd_data)
        wav_e = wav_b + len(snd_mu)
        snd_data.resize(wav_e)
        snd_data[wav_b:wav_e] = snd_mu
        samples.append(
                SpokenSample(
                    voice_index=speaker_id_map[voice_id], wav_b=wav_b,
                    wav_e=wav_e, file_path=snd_path
                    )
                )
        if len(samples) % 100 == 0:
            print('Converted {} files of {}.'.format(len(samples),
                len(catalog), file=stderr))
            stderr.flush()

    with open(dat_file, 'wb') as dat_fh:
        state = {
                'samples': samples,
                'snd_dtype': snd_dtype,
                'snd_data': snd_data
                }
        pickle.dump(state, dat_fh)
        


SpokenSample = namedtuple('SpokenSample', [
    'voice_index',   # index of the speaker for this sample
    'wav_b',         # start position of sample in full wav data buffer
    'wav_e',         # end position of sample in full wav data buffer
    'file_path'      # path to .wav file for this sample
    ]
    )



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

    def populate(self):
        """
        sets the data for one sample in the batch
        """
        rg = torch.empty((self.ds.batch_size), dtype=torch.int64).cpu()
        picks = rg.random_() % len(self.ds.in_start)
        nz = self.ds.emb_len
        wav_trim = self.ds.trim_dec_in

        for b, wi in enumerate(picks):
            s, voice_ind = self.ds.in_start[wi]
            wav_enc_input = self.ds.snd_data[s:s + self.ds.enc_in_len]
            self.wav_dec_input[b,...] = wav_enc_input[wav_trim[0]:wav_trim[1]]
            self.mel_enc_input[b,...] = self.ds.mfcc_proc.func(wav_enc_input)
            self.voice_index[b] = voice_ind 
            self.jitter_index[b,:] = \
                    torch.tensor(self.ds.jitter.gen_indices(nz) + b * nz) 
        assert self.wav_dec_input.shape[0] == 8


    def to(self, device):
        shape_cpu = self.voice_index.shape
        self.voice_index = self.voice_index.to(device)
        shape_tpu = self.voice_index.shape
        assert shape_cpu == shape_tpu
        self.jitter_index = self.jitter_index.to(device)
        self.wav_dec_input = self.wav_dec_input.to(device)
        self.mel_enc_input = self.mel_enc_input.to(device)
        assert self.wav_dec_input.shape[0] == 8



class Slice(torch.utils.data.IterableDataset):
    """
    Defines the current batch of data in iterator style.
    Use with automatic batching disabled, and collate_fn = lambda x: x
    """
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


