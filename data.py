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
    mel_data = np.empty((0), dtype=snd_dtype)
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
        
    
def make_mel():
    mfcc_proc = mfcc.ProcessWav(sample_rate, win_sz, hop_sz, n_mels, n_mfcc)

    # mel: C, T  (n_mels, n_timesteps)
    # reshape to T, C for ease in slicing timesteps 
    # then flatten
    # we must revert the shape of slices back to C, T
    mel = mfcc_proc.func(snd)
    if n_mel_chan is None:
        n_mel_chan = mel.shape[0]
    n_mel_elem = mel.shape[1]

    mel = mel.transpose((1, 0)).flatten()
    mel_raw_b = len(mel_data)
    mel_raw_e = mel_raw_b + len(mel)
    mel_data.resize(mel_raw_e)
    mel_data[mel_raw_b:mel_raw_e] = mel
    assert mel_raw_b % n_mel_chan == 0
    assert mel_raw_e % n_mel_chan == 0
    mel_b = mel_raw_b // n_mel_chan
    mel_e = mel_raw_e // n_mel_chan



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
        self.voice_index = torch.empty(self.ds.batch_size, dtype=torch.long)
        self.jitter_index = torch.empty(self.ds.batch_size, dataset.emb_len, dtype=torch.long)
        self.wav_input = torch.empty(self.ds.batch_size, wav_len)
        self.mel_input = torch.empty(self.ds.batch_size, mel_chan, mel_len) 

    def __repr__(self):
        fmt = (
            'voice_index: {}\n' + 
            'jitter_index: {}\n' + 
            'wav_input.shape: {}\n' + 
            'mel_input.shape: {}\n'
        )
        return fmt.format(self.voice_index, self.jitter_index,
                self.wav_input.shape, self.mel_input.shape)

    def populate(self):
        """
        sets the data for one sample in the batch
        """
        rg = torch.empty((self.ds.batch_size), dtype=torch.int64).cpu()
        picks = rg.random_()[0] % len(self.in_start)
        nz = self.ds.max_embed_len

        for b, wi in enumerate(picks):
            s, voice_ind = self.in_starts[wi]
            self.wav_input[b,...] = self.ds.snd_data[s:s + self.ds.enc_in_len]
            self.mel_input[b,...] = self.ds.mfcc_func(self.wav_input[b,...])
            self.voice_index[b] = voice_ind 
            self.jitter_index[b,:] = \
                    torch.tensor(self.ds.jitter.gen_indices(nz) + b * nz) 


    def to(self, device):
        self.voice_index = self.voice_index.to(device)
        self.jitter_index = self.jitter_index.to(device)
        self.wav_input = self.wav_input.to(device)
        self.mel_input = self.mel_input.to(device)



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

    def load_data(self, dat_file):
        try:
            with open(dat_file, 'rb') as dat_fh:
                dat = pickle.load(dat_fh)
        except IOError:
            print('Could not open preprocessed data file {}.'.format(
                dat_file), file=stderr)
            stderr.flush()
            exit(1)

        mfcc_pars = dat['mfcc_params']
        self.samples = dat['samples']
        self.n_mel_chan = mfcc_pars['n_mel_chan']

        self._load_sample_data(dat['snd_data'], dat['mel_data'],
                dat['snd_dtype'], self.n_mel_chan)

        self.mfcc_vc = vconv.VirtualConv(
                filter_info=mfcc_pars['window_size'], stride=mfcc_pars['hop_size'],
                parent=None, name='MFCC'
        )


    def __setstate__(self, init_args):
        self.init_args = init_args 
        self._initialize()


    def __getstate__(self):
        return self.init_args


    def num_speakers(self):
        return len(set(map(lambda s: s.voice_index, self.samples)))


    def init_geometry(self):
        """
        Initializes:
        self.enc_in_len
        self.trim_ups_out
        self.trim_dec_out
        self.trim_dec_in
        """
        # Calculate max length of mfcc encoder input and wav decoder input
        w = self.window_batch_size
        beg_grcc_vc = self.decoder_vcs['beg_grcc']
        end_grcc_vc = self.decoder_vcs['end_grcc']
        end_ups_vc = self.decoder_vcs['last_upsample']

        do = GridRange((0, 100000), (0, w), 1)
        di = vconv.input_range(beg_grcc_vc, end_grcc_vc, do)
        ei = vconv.input_range(self.mfcc_vc, end_grcc_vc, do)
        uo = vconv.output_range(self.mfcc_vc, end_ups_vc, ei)

        # Needed for trimming various tensors
        self.enc_in_len = ei.sub_length()
        self.trim_dec_in = di.sub[0] - ei.sub[0], di.sub[1] - ei.sub[0]
        self.trim_ups_out = uo.sub[0] - di.sub[0], uo.sub[1] - di.sub[0]
        self.trim_dec_out = do.sub[0] - ei.sub[0], do.sub[1] - ei.sub[0]

        # Generate slices from input
        self.in_start = []
        for si, sam in enumerate(self.samples):
            for b in range(sam.wav_b, sam.wav_e - w, w):
                self.in_start.append((b, si))


    def post_init(self, encoder_vcs, decoder_vcs):
        """
        Initializes:
        self.slices
        Depends on information computed from the model, so must be
        called after model construction.
        """
        self.encoder_vcs = encoder_vcs
        self.decoder_vcs = decoder_vcs
        self.init_geometry()


    def _load_sample_data(self, snd_np, mel_np, snd_dtype, n_mel_chan):
        """
        Populates self.snd_data and self.mel_data
        """
        if snd_dtype is np.uint8:
            snd_data = torch.ByteTensor(snd_np)
        elif snd_dtype is np.uint16:
            snd_data = torch.ShortTensor(snd_np)
        elif snd_dtype is np.int32:
            snd_data = torch.IntTensor(snd_np)

        # shape: T, M
        mel_data = torch.FloatTensor(mel_np).reshape((-1, n_mel_chan))

        self.snd_data = snd_data
        self.mel_data = mel_data

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
        vb.mel_input.detach_()
        vb.mel_input.requires_grad_(False)
        vb.populate()

        if self.target_device:
            vb.to(self.target_device)
        vb.mel_input.requires_grad_(True)

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


