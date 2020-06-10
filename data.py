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
    def __init__(self, dataset, start_epoch=0, start_step=0):
        super().__init__(dataset)
        self.dataset = dataset
        self.epoch = start_epoch
        self.step = start_step

    def __iter__(self):
        def _gen():
            while True:
                n = len(self.dataset)
                indices = t.randperm(n).tolist()
                for i in indices:
                    self.step += 1
                    yield i
                self.epoch += 1
                self.step = 0
        return _gen()

    def __len__(self):
        return int(1e20)

    def set_pos(self, epoch, step):
        self.epoch = epoch
        self.step = step



def load_data(dat_file):
    try:
        with open(dat_file, 'rb') as dat_fh:
            dat = pickle.load(dat_fh)
    except IOError:
        print(f'Could not open preprocessed data file {dat_file}.', file=stderr)
        stderr.flush()
    return dat

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

    
class DataProcessor():
    def __init__(self, hps, dat_file, mfcc_func, slice_size, train_mode,
            start_epoch=0, start_step=0):
        super().__init__()
        self.jitter = jitter.Jitter(hps.jitter_prob) 
        self.mfcc = mfcc_func

        def collate_fn(batch):
            wav = t.stack([t.from_numpy(b[0]) for b in batch]).float()
            mel = t.stack([t.from_numpy(self.mfcc(b[0])) for b in
                batch]).float()
            voice = t.tensor([b[1] for b in batch]).long()
            jitter = t.stack([t.from_numpy(self.jitter(mel.size()[2])) for _ in
                range(len(batch))]).long()

            if not train_mode:
                paths = [b[2] for b in batch]
                return wav, mel, voice, jitter, paths
            else:
                return wav, mel, voice, jitter

        if train_mode:
            self.dataset = SliceDataset(slice_size, hps.n_win_batch)
            self.dataset.load_data(dat_file)
            self.sampler = LoopingRandomSampler(self.dataset, start_epoch,
                    start_step)
            self.loader = DataLoader(self.dataset, sampler=self.sampler,
                    batch_size=hps.n_batch, pin_memory=False,
                    collate_fn=collate_fn)
        else:
            self.dataset = WavFileDataset()
            self.dataset.load_data(dat_file)
            self.sampler = SequentialSampler(self.dataset)
            self.loader = DataLoader(self.dataset, batch_size=1,
                    sampler=self.sampler, pin_memory=False, drop_last=False,
                    collate_fn=collate_fn)

    @property
    def epoch(self):
        return self.sampler.epoch

    @property
    def step(self):
        return self.sampler.step

    @property
    def global_step(self):
        return len(self.dataset) * self.epoch + self.step



