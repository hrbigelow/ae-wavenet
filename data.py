# Preprocess Data
from sys import stderr, exit
import pickle
import numpy as np
import torch as t
from torch.utils.data import Dataset, DataLoader, Sampler
import jitter
from torch import nn
import vconv
import copy
import parse_tools
from hparams import setup_hparams
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
                    voice_index=speaker_id_map[voice_id],
                    wav_b=wav_b, wav_e=wav_e,
                    # mel_b=mel_b, mel_e=mel_e, 
                    file_path=snd_path
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


class LoopingRandomSampler(Sampler):
    def __init__(self, dataset, num_replicas=1, rank=0):
        super().__init__(dataset)
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        print(f'LoopingRandomSampler with {self.rank} out of {self.num_replicas}', file=stderr)

    def __iter__(self):
        def _gen():
            while True:
                n = len(self.dataset)
                vals = list(range(self.rank, n, self.num_replicas))
                perms = t.randperm(len(vals)).tolist()
                indices = [vals[i] for i in perms]
                for i in indices:
                    yield i

        return _gen()

    def __len__(self):
        return int(2**31)




def load_data(dat_file):
    try:
        with open(dat_file, 'rb') as dat_fh:
            dat = pickle.load(dat_fh)
    except IOError:
        print(f'Could not open preprocessed data file {dat_file}.', file=stderr)
        stderr.flush()
    return dat

class TrackerDataset(Dataset):
    """
    Tracks and provides the epoch and step.
    If using with replicas and a subsetting sampler that samples
    1/sampling_freq of the dataset
    """
    def __init__(self, dataset, start_epoch=0, start_step=0, sampling_freq=1):
        self.dataset = dataset
        self.epoch = start_epoch
        self.step = start_step
        self.sampling_freq = sampling_freq
        self.len = None

    def __len__(self):
        if self.len is None:
            self.len = len(self.dataset)
        return self.len

    def __getitem__(self, item):
        self.step += self.sampling_freq 
        if self.step >= len(self):
            self.epoch += 1
            self.step = 0
        return self.dataset[item], self.epoch, self.step

    def set_pos(self, epoch, step):
        self.epoch = epoch
        self.step = step


class SliceDataset(Dataset):
    """
    Return slices of wav files of fixed size
    """
    def __init__(self, slice_size, n_win_batch):
        self.slice_size = slice_size
        self.n_win_batch = n_win_batch 
        self.in_start = []


    def load_data(self, dat_file):
        dat = load_data(dat_file)
        self.samples = dat['samples']
        self.snd_data = dat['snd_data'].astype(dat['snd_dtype'])

        w = self.n_win_batch
        for sam in self.samples:
            for b in range(sam.wav_b, sam.wav_e - self.slice_size, w):
                self.in_start.append((b, sam.voice_index))

    def num_speakers(self):
        ns = max(s.voice_index for s in self.samples) + 1
        return ns

    def __len__(self):
        return len(self.in_start)

    def __getitem__(self, item):
        s, voice_ind = self.in_start[item]
        return self.snd_data[s:s + self.slice_size], voice_ind



class WavFileDataset(Dataset):
    """
    Returns entire wav files
    """
    def __init__(self, hps):
        super().__init__()

    def load_data(self, dat_file):
        dat = load_data(dat_file)
        self.samples = dat['samples']
        self.snd_data = dat['snd_data'].astype(dat['snd_dtype'])

    def num_speakers(self):
        ns = max(s.voice_index for s in self.samples) + 1
        return ns
    
    def __len__(self):
        return len(self.samples) 

    def __getitem__(self, item):
        sam = ds.samples[item]
        return (self.snd_data[sam.wav_b:sam.wav_e],
                sam.voice_index,
                sam.file_path)


class Collate():
    def __init__(self, mfcc, jitter, train_mode):
        self.train_mode = train_mode
        self.mfcc = mfcc
        self.jitter = jitter

    def __call__(self, batch):
        data = [b[0] for b in batch]

        # epoch, step
        position = t.tensor(batch[-1][1:])

        wav = t.stack([t.from_numpy(d[0]) for d in data]).float()
        mel = t.stack([t.from_numpy(self.mfcc(d[0])) for d in
            data]).float()
        voice = t.tensor([d[1] for d in data]).long()
        jitter = t.stack([t.from_numpy(self.jitter(mel.size()[2])) for _ in
            range(len(data))]).long()

        if self.train_mode:
            return wav, mel, voice, jitter, position 
        else:
            paths = [b[0][2] for b in batch]
            return wav, mel, voice, jitter, paths, position


class DataProcessor():
    def __init__(self, hps, dat_file, mfcc_func, slice_size, train_mode,
            start_epoch=0, start_step=0, num_replicas=1, rank=0):
        super().__init__()
        jitter_func = jitter.Jitter(hps.jitter_prob) 

        train_collate_fn = Collate(mfcc_func, jitter_func, train_mode=True)
        test_collate_fn = Collate(mfcc_func, jitter_func, train_mode=False)

        if train_mode:
            slice_dataset = SliceDataset(slice_size, hps.n_win_batch)
            slice_dataset.load_data(dat_file)
            stderr.flush()
            self.dataset = TrackerDataset(slice_dataset, start_epoch,
                    start_step, sampling_freq=num_replicas)
            self.sampler = LoopingRandomSampler(self.dataset, num_replicas, rank)
            self.loader = DataLoader(self.dataset, sampler=self.sampler,
                    # If set >0, multiprocessing is used, which prevents
                    # getting accurate position information
                    num_workers=hps.n_loader_workers,
                    batch_size=hps.n_batch, pin_memory=False,
                    collate_fn=train_collate_fn)
        else:
            wav_dataset = WavFileDataset()
            wav_dataset.load_data(dat_file)
            self.dataset = TrackerDataset(wav_dataset, 0, 0)
            self.sampler = SequentialSampler(self.dataset)
            self.loader = DataLoader(self.dataset, batch_size=1,
                    sampler=self.sampler, pin_memory=False, drop_last=False,
                    collate_fn=test_collate_fn)

    @property
    def global_step(self):
        return len(self.dataset) * self.dataset.epoch + self.dataset.step

    """
    @property
    def epoch(self):
        return self.dataset.epoch

    @property
    def step(self):
        return self.dataset.step
    """



