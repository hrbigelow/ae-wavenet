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
        ds = dataset
        self.voice_index = torch.empty((ds.batch_size,), dtype=torch.long)
        self.jitter_index = torch.empty((ds.batch_size, ds.embed_len),
                dtype=torch.long)
        self.wav_dec_input = torch.empty((ds.batch_size, ds.dec_in_len))
        self.mel_enc_input = torch.empty((ds.batch_size, ds.num_mel_chan(),
            ds.enc_in_mel_len)) 

    def __repr__(self):
        fmt = (
            'voice_index: {}\n' + 
            'jitter_index: {}\n' + 
            'wav_dec_input.shape: {}\n' + 
            'mel_enc_input.shape: {}\n'
        )
        return fmt.format(self.voice_index, self.jitter_index,
                self.wav_dec_input.shape, self.mel_enc_input.shape)

    def populate(self, dataset):
        """
        sets the data for one sample in the batch
        """
        ds = dataset
        rg = torch.empty((ds.batch_size), dtype=torch.int64).cpu()
        picks = rg.random_() % len(ds.in_start) 
        nz = ds.embed_len
        trim = ds.trim_dec_in

        for b, wi in enumerate(picks):
            s, voice_ind = ds.in_start[wi]
            wav_enc_input = ds.snd_data[s:s + ds.enc_in_len]
            # print('wav_enc_input.shape: {}, s: {}, ds.snd_data.shape: {}'.format(
            #     wav_enc_input.shape, s, ds.snd_data.shape), file=stderr)
            # stderr.flush()

            self.wav_dec_input[b,...] = wav_enc_input[trim[0]:trim[1]]
            self.mel_enc_input[b,...] = ds.mfcc_proc.func(wav_enc_input)
            self.voice_index[b] = voice_ind 
            self.jitter_index[b,:] = \
                    torch.tensor(ds.jitter.gen_indices(nz) + b * nz) 
        self.mel_enc_input /= \
            self.mel_enc_input.std(dim=(1,2)).unsqueeze(1).unsqueeze(1)


    def to(self, device):
        self.voice_index = self.voice_index.to(device)
        self.jitter_index = self.jitter_index.to(device)
        self.wav_dec_input = self.wav_dec_input.to(device)
        self.mel_enc_input = self.mel_enc_input.to(device)



class Slice(torch.utils.data.IterableDataset):
    """
    Defines the current batch of data in iterator style.
    Use with automatic batching disabled, and collate_fn = lambda x: x
    """
    def __init__(self, batch_size, window_batch_size, jitter_prob,
            sample_rate, mfcc_win_sz, mfcc_hop_sz, n_mels, n_mfcc):
        self.init_args = {
                'batch_size': batch_size,
                'window_batch_size': window_batch_size,
                'jitter_prob': jitter_prob,
                'sample_rate': sample_rate,
                'mfcc_win_sz': mfcc_win_sz,
                'mfcc_hop_sz': mfcc_hop_sz,
                'n_mels': n_mels,
                'n_mfcc': n_mfcc
                }
        self._initialize()


    def _initialize(self):
        super(Slice, self).__init__()
        self.target_device = None
        self.__dict__.update(self.init_args)
        self.jitter = jitter.Jitter(self.jitter_prob) 
        self.mfcc_proc = mfcc.ProcessWav(
                sample_rate=self.sample_rate,
                win_sz=self.mfcc_win_sz,
                hop_sz=self.mfcc_hop_sz,
                n_mels=self.n_mels,
                n_mfcc=self.n_mfcc)
        self.mfcc_vc = vconv.VirtualConv(filter_info=self.mfcc_win_sz,
                stride=self.mfcc_hop_sz, parent=None, name='MFCC')

    def load_data(self, dat_file):
        try:
            with open(dat_file, 'rb') as dat_fh:
                dat = pickle.load(dat_fh)
        except IOError:
            print('Could not open preprocessed data file {}.'.format(
                dat_file), file=stderr)
            stderr.flush()
            exit(1)

        self.samples = dat['samples']
        self._load_sample_data(dat['snd_data'], dat['snd_dtype'])


    def __setstate__(self, init_args):
        self.init_args = init_args 
        self._initialize()

    def __getstate__(self):
        return self.init_args


    def num_speakers(self):
        return max(map(lambda s: s.voice_index, self.samples)) + 1

    def num_mel_chan(self):
        return self.mfcc_proc.n_out


    def post_init(self, model):
        self.trim_dec_in = model.trim_dec_in.numpy()
        self.embed_len = model.embed_len
        self.enc_in_len = model.enc_in_len
        self.dec_in_len = model.dec_in_len
        self.enc_in_mel_len = model.enc_in_mel_len
        
        w = self.window_batch_size
        self.in_start = []
        for sam in self.samples:
            for b in range(sam.wav_b, sam.wav_e - self.enc_in_len, w):
                self.in_start.append((b, sam.voice_index))


    def _load_sample_data(self, snd_np, snd_dtype):
        """
        Populates self.snd_data
        """
        if snd_dtype is np.uint8:
            snd_data = torch.ByteTensor(snd_np)
        elif snd_dtype is np.uint16:
            snd_data = torch.ShortTensor(snd_np)
        elif snd_dtype is np.int32:
            snd_data = torch.IntTensor(snd_np)
        self.snd_data = snd_data


    def set_target_device(self, target_device):
        self.target_device = target_device


    def __iter__(self):
        return self

    def __next__(self):
        """
        Get a random slice of a file, together with its start position and ID.
        Random state is from torch.{get,set}_rng_state().  It is on the CPU,
        not GPU.
        """
        vb = VirtualBatch(self)
        vb.mel_enc_input.detach_()
        vb.mel_enc_input.requires_grad_(False)
        vb.populate(self)

        if self.target_device:
            vb.to(self.target_device)
        vb.mel_enc_input.requires_grad_(True)

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


