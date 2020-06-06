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
import parse_tools
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



class VirtualBatch(object):
    def __init__(self, dataset):
        import random

        super(VirtualBatch, self).__init__()
        ds = dataset
        self.voice_idx = torch.empty((ds.batch_size,), dtype=torch.long)
        self.jitter_idx = torch.empty((ds.batch_size, ds.embed_len),
                dtype=torch.long)
        self.wav = torch.empty((ds.batch_size, ds.enc_in_len))
        self.mel = torch.empty((ds.batch_size, ds.num_mel_chan(),
            ds.enc_in_mel_len)) 
        self.wav_rng = [None] * ds.batch_size
        self.mel_rng = [None] * ds.batch_size

        self.picks = list(range(len(ds.in_start)))
        random.shuffle(self.picks)
        self.pos = 0
        

    def __repr__(self):
        fmt = (
            'voice_idx: {}\n' + 
            'jitter_idx: {}\n' + 
            'wav.shape: {}\n' + 
            'mel.shape: {}\n' +
            'wav_rng: {}\n' +
            'mel_rng: {}\n'
        )
        return fmt.format(self.voice_idx, self.jitter_idx,
                self.wav.shape, self.mel.shape,
                self.wav_rng, self.mel_rng)

    def populate(self, dataset):
        """
        sets the data for one sample in the batch
        """
        ds = dataset
        rg = torch.empty((ds.batch_size), dtype=torch.int64).cpu()

        nz = ds.embed_len

        for b in range(ds.batch_size):
            wi = self.picks[self.pos]
            s, voice_ind = ds.in_start[wi]
            _wav = ds.snd_data[s:s + ds.enc_in_len]
            # print('wav_enc_input.shape: {}, s: {}, ds.snd_data.shape: {}'.format(
            #     wav_enc_input.shape, s, ds.snd_data.shape), file=stderr)
            # stderr.flush()
            self.wav_rng[b] = s, s + ds.enc_in_len 
            self.mel_rng[b] = s, s + ds.enc_in_len
            self.wav[b,...] = _wav
            self.mel[b,...] = ds.mfcc_proc.func(_wav)
            self.voice_idx[b] = voice_ind 
            self.jitter_idx[b,:] = \
                    torch.tensor(ds.jitter.gen_indices(nz) + b * nz) 

            # loop indefinitely
            self.pos = (self.pos + 1) % len(ds.in_start)

        # self.mel /= \
        #    self.mel.std(dim=(1,2)).unsqueeze(1).unsqueeze(1)


    def to(self, device):
        self.voice_idx = self.voice_idx.to(device)
        self.jitter_idx = self.jitter_idx.to(device)
        self.wav = self.wav.to(device)
        self.mel = self.mel.to(device)

    def validate(self, ds):
        """
        validates that the internal coordinates match those directly fetched
        """
        for b in range(ds.batch_size):
            fetched_wav, fetched_mel = ds.fetch_at(*self.wav_rng[b],
                    *self.mel_rng[b])
            assert torch.equal(fetched_wav.float(), self.wav[b].cpu())
            assert torch.equal(fetched_mel.float(), self.mel[b].cpu())



class MfccBatch(object):
    """
    Yield a batch of wav and accompanying mfcc input
    from a full, continuous recording.
    - Given the varying lengths of different recordings,
    - this can only handle a batch size of 1
    """
    def __init__(self, dataset):
        super(MfccBatch, self).__init__()
        ds = dataset
        assert ds.batch_size == 1, 'Mfcc only supports batch size 1'
        self.pos = 0
        self.valid = False
        self.voice_idx = torch.empty((ds.batch_size,), dtype=torch.long)

    def populate(self, dataset):
        ds = dataset
        self.valid = False
        if self.pos == len(ds.samples):
            return

        sam = ds.samples[self.pos]
        __, self.voice_idx[0] = ds.in_start[self.pos] 
        self.wav_enc_rng = sam.wav_b, sam.wav_e
        self.mel_enc_rng = sam.wav_b, sam.wav_e

        self.wav = ds.snd_data[sam.wav_b:sam.wav_e].unsqueeze(0)
        _mel = ds.mfcc_proc.func(self.wav[0]).unsqueeze(0)
        # _mel /= _mel.std(dim=(1,2)).unsqueeze(1).unsqueeze(1)
        self.mel = _mel.type(torch.float32)
        embed_len = self.mel.size()[2]
        self.jitter_idx = torch.tensor(ds.jitter.gen_indices(embed_len),
                dtype=torch.long)
        self.file_path = sam.file_path
        self.valid = True
        self.pos += 1


    def to(self, device):
        self.mel = self.mel.to(device)
        self.wav = self.wav.to(device)
        self.voice_idx = self.voice_idx.to(device)
        self.jitter_idx = self.jitter_idx.to(device)

    def validate(self, ds):
        fetched_wav, fetched_mel = ds.fetch_at(*self.wav_enc_rng,
                *self.mel_enc_rng)
        assert torch.equal(fetched_wav.byte(), self.wav[0].cpu())
        assert torch.equal(fetched_mel.float(), self.mel[0].cpu())


class Slice(torch.utils.data.IterableDataset):
    """
    Defines the current batch of data in iterator style.
    Use with automatic batching disabled, and collate_fn = lambda x: x
    """
    def __init__(self, opts):
        opts_dict = vars(opts)
        pre_pars = parse_tools.get_prefixed_items(opts_dict, 'pre_')
        self.init_args = {
                'batch_size': opts.n_batch,
                'window_batch_size': opts.n_win_batch,
                'jitter_prob': opts.jitter_prob,
                'sample_rate': pre_pars['sample_rate'],
                'mfcc_win_sz': pre_pars['mfcc_win_sz'],
                'mfcc_hop_sz': pre_pars['mfcc_hop_sz'],
                'n_mels': pre_pars['n_mels'],
                'n_mfcc': pre_pars['n_mfcc']
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
        ns = max(s.voice_index for s in self.samples) + 1
        return ns

    def num_mel_chan(self):
        return self.mfcc_proc.n_out

    def override(self, n_batch=None, n_win_batch=None):
        """
        override values from checkpoints
        """
        if n_batch is not None:
            self.batch_size = n_batch
        if n_win_batch is not None:
            self.window_batch_size = n_win_batch


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
        self.vb = VirtualBatch(self)
        return self

    def __next__(self):
        """
        Get a random slice of a file, together with its start position and ID.
        Random state is from torch.{get,set}_rng_state().  It is on the CPU,
        not GPU.
        """
        self.vb.mel.detach_()
        self.vb.mel.requires_grad_(False)
        self.vb.populate(self)

        if self.target_device:
            self.vb.to(self.target_device)
        self.vb.mel.requires_grad_(True)

        return self.vb 

    def fetch_at(self, wav_b, wav_e, wav_mel_b, wav_mel_e):
        """
        fetches specific slice of wav data, and its matching mel data, directly
        """
        wav = self.snd_data[wav_b:wav_e]
        mel = self.mfcc_proc.func(self.snd_data[wav_mel_b:wav_mel_e])
        return wav, mel
        # return wav.to(self.target_device), mel.to(self.target_device)


class MfccInference(Slice):
    """
    The data iterator for training the mfcc inverter model.
    Yields MfccBatch
    """
    def __init__(self, slice_dataset):
        self.__dict__.update(vars(slice_dataset))
        self.batch_size = 1
        self.mb = MfccBatch(self)

    def __next__(self):
        """
        Get a random slice of a file, together with its start position and ID.
        Random state is from torch.{get,set}_rng_state().  It is on the CPU,
        not GPU.
        """
        # mb.mel_enc_input.detach_()
        # mb.mel_enc_input.requires_grad_(False)
        self.mb.populate(self)
        if not self.mb.valid:
            raise StopIteration

        if self.target_device:
            self.mb.to(self.target_device)
        # mb.mel_enc_input.requires_grad_(True)

        return self.mb 


class WavLoader(torch.utils.data.DataLoader):
    """
    - Wrapper to convert a Slice to a DataLoader.
    - May be wrapped with torch_xla.distributed.parallel_loader.
    - returns batches of tensors on cpu, optionally pushing them to
    - target_device if provided
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



