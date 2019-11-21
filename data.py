# Preprocess Data
from sys import stderr, exit
import pickle
import librosa
import numpy as np
import torch
import torch.utils.data
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

def convert(catalog, dat_file, n_quant, sample_rate=16000, win_sz=400, hop_sz=160,
        n_mels=80, n_mfcc=13):
    """
    Convert all input data and save a dat file 
    """
    mfcc_proc = mfcc.ProcessWav(sample_rate, win_sz, hop_sz, n_mels, n_mfcc)

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
        samples.append(
                SpokenSample(
                    voice_index=speaker_id_map[voice_id], wav_b=wav_b,
                    wav_e=wav_e, mel_b=mel_b, mel_e=mel_e,
                    file_path=snd_path
                    )
                )
        if len(samples) % 100 == 0:
            print('Converted {} files of {}.'.format(len(samples),
                len(catalog), file=stderr))
            stderr.flush()

    with open(dat_file, 'wb') as dat_fh:
        state = {
                'mfcc_params': {
                    'n_mel_chan': n_mel_chan,
                    'n_quant': n_quant,
                    'window_size': win_sz,
                    'hop_size': hop_sz,
                    'n_mels': n_mels,
                    'n_mfcc': n_mfcc
                    },
                'samples': samples,
                'snd_dtype': snd_dtype,
                'snd_data': snd_data,
                'mel_data': mel_data
                }
        pickle.dump(state, dat_fh)
        
        


# Represents a slice of a sample plus the geometry
SampleSlice = namedtuple('SampleSlice', [
    'wav_offset',    # Offset into full wav dataset
    'mel_offset',    # Offset into full mel dataset
    'dec_wav_slice', # wav slice input to decoder, relative to wav_offset
    'mel_in_slice',  # mel slice input to encoder, relative to mel_offset
    'loss_wav_slice',# slice of wav decoder, relative to dec_wav_slice 
    'lcond_slice',   # slice of lcond tensor, absolute
    'voice_index'    # index of the speaker for this sample
    ]
    )

SpokenSample = namedtuple('SpokenSample', [
    'voice_index',   # index of the speaker for this sample
    'wav_b',         # start position of sample in full wav data buffer
    'wav_e',         # end position of sample in full wav data buffer
    'mel_b',         # start position of sample in full mel data buffer
    'mel_e',         # end position of sample in full mel data buffer
    'file_path'      # path to .wav file for this sample
    ]
    )


OutputRange = namedtuple('OutputRange', [
    'output_gr',    # grid range of the output slice
    'sample_index'  # index of the SpokenSample this is from
    ]
    )


class VirtualBatch(object):
    def __init__(self, batch_size, max_wav_len, max_mel_len, mel_chan):
        super(VirtualBatch, self).__init__()
        self.batch_size = batch_size
        self.voice_index = torch.empty(batch_size, dtype=torch.long)
        self.lcond_slice = torch.empty(batch_size, max_wav_len,
                dtype=torch.long)
        self.loss_wav_slice = [None] * batch_size 
        self.wav_input = torch.empty(batch_size, max_wav_len)
        self.mel_input = torch.empty(batch_size, mel_chan, max_mel_len) 

    def __repr__(self):
        fmt = ('voice_index: {}\nlcond_slice: {}\nloss_wav_slice: {}\n' +
                'wav_input.shape: {}\nmel_input.shape: {}\n')
        return fmt.format(self.voice_index, self.lcond_slice,
                self.loss_wav_slice, self.wav_input.shape,
                self.mel_input.shape)

    def set_one(self, b, sample_slice, data_source):
        """
        sets the data for one sample in the batch
        """
        ss = sample_slice
        wo = ss.wav_offset
        mo = ss.mel_offset
        dws = ss.dec_wav_slice
        mis = ss.mel_in_slice

        self.voice_index[b] = ss.voice_index
        offset = b * data_source.max_lcond_len
        self.lcond_slice[b,:] = torch.arange(offset + ss.lcond_slice[0],
                offset + ss.lcond_slice[1])
        self.loss_wav_slice[b] = ss.loss_wav_slice 
        self.wav_input[b,...] = data_source.snd_data[wo + dws[0]:wo + dws[1]] 
        self.mel_input[b,...] = data_source.mel_data[mo + mis[0]:mo +
                mis[1],:].transpose(1, 0)

        # self.wav_input[b,...] = data_source.snd_data[3184397:3186543]
        # self.mel_input[b,...] = \
        #         data_source.mel_data[19855:19899,:].transpose(1, 0)

    def to(self, device):
        self.voice_index = self.voice_index.to(device)
        self.lcond_slice = self.lcond_slice.to(device)
        self.wav_input = self.wav_input.to(device)
        self.mel_input = self.mel_input.to(device)


    def valid(self):
        lw_len = self.loss_wav_len()
        return (
                all(map(lambda lw: lw[1] - lw[0] == lw_len,
                    self.loss_wav_slice))
                )

    def lcond_len(self):
        return self.lcond_slice.size()[1]

    def loss_wav_len(self):
        return self.loss_wav_slice[0][1] - self.loss_wav_slice[0][0]


class Slice(torch.utils.data.IterableDataset):
    """
    Defines the current batch of data in iterator style.
    Use with automatic batching disabled, and collate_fn = lambda x: x
    """
    def __init__(self, dat_file, batch_size, window_batch_size):
        self.init_args = {
                'dat_file': dat_file,
                'batch_size': batch_size,
                'window_batch_size': window_batch_size
                }
        self._initialize()


    def _initialize(self):
        """
        Sets
        self.batch_size
        self.window_batch_size
        self.mfcc_vc
        self.snd_data
        self.mel_data
        """
        super(Slice, self).__init__()
        self.target_device = None
        self.dat_file = self.init_args['dat_file']
        self.batch_size = self.init_args['batch_size']
        self.window_batch_size = self.init_args['window_batch_size']

        try:
            with open(self.dat_file, 'rb') as dat_fh:
                dat = pickle.load(dat_fh)
        except IOError:
            print('Could not open preprocessed data file {}.'.format(self.dat_file),
                    file=stderr)
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
        self.max_wav_len
        self.max_mel_len
        self.max_lcond_len
        """
        # Calculate max length of mfcc encoder input and wav decoder input
        w = self.window_batch_size
        beg_grcc_vc = self.vcs['beg_grcc']
        end_grcc_vc = self.vcs['end_grcc']
        autoenc = self.mfcc_vc, end_grcc_vc
        autoenc_clip = self.mfcc_vc.child, end_grcc_vc
        enc = self.mfcc_vc.child, self.vcs['last_upsample']
        dec = beg_grcc_vc, end_grcc_vc 

        max_spacing = vconv.max_spacing(*autoenc, 1)
        max_mel_len = 0
        max_lcond_len = 0
        for b in range(max_spacing):
            out = vconv.GridRange((0, 100000), (b, b + w), 1)
            mfcc = vconv.input_range(*autoenc_clip, out)
            # print(mfcc.sub_length(), end=' ')
            max_mel_len = max(mfcc.sub_length(), max_mel_len)
            max_mel_in_gr = vconv.GridRange((0, 1000000), (b, b +
                    max_mel_len * mfcc.gs), mfcc.gs)
            lcond_gr = vconv.output_range(*enc, max_mel_in_gr)
            max_lcond_len = max(max_lcond_len, lcond_gr.sub_length())
        
        # Calculate decoder wav input length
        slice_out = vconv.GridRange((0, 100000), (0, w), 1)
        self.max_wav_len = vconv.input_range(*dec, slice_out).sub_length()
        self.max_mel_len = max_mel_len
        self.max_lcond_len = max_lcond_len
        

    def init_slices(self):
        """
        Initialize:
        self.slices
        """
        # generate all slices
        win_size = self.window_batch_size
        self.out_range = []
        for si in range(len(self.samples)): 
            self._add_out_ranges(si, win_size)


    def _add_out_ranges(self, si, win_size):
        """
        Initialize self.out_range
        """
        preproc = self.mfcc_vc, self.mfcc_vc
        autoenc = self.mfcc_vc, self.vcs['end_grcc']
        autoenc_clip = self.mfcc_vc.child, self.vcs['end_grcc']

        wlen = self.samples[si].wav_e - self.samples[si].wav_b
        full_wav_in = vconv.GridRange((0, wlen), (0, wlen), 1)
        full_mel_in = vconv.output_range(*preproc, full_wav_in)
        full_out = vconv.output_range(*autoenc, full_wav_in) 
        assert full_out.gs == 1
        slice_out = vconv.GridRange(full_out.full, 
                (full_out.sub[1] - win_size, full_out.sub[1]), full_out.gs)
        while slice_out.valid():
            slice_out = vconv.GridRange(slice_out.full,
                    (slice_out.sub[0] - win_size,
                    slice_out.sub[1] - win_size),
                    slice_out.gs)
            self.out_range.append(OutputRange(slice_out, si))


    def calc_slice(self):
        """
        Return a SampleSlice corresponding to self.out_range[oi] 
        """
        rg = torch.empty((1), dtype=torch.int64).cpu()
        while True:
            pick = rg.random_()[0] % len(self.out_range)
            out_range = self.out_range[pick]
            slice_out = out_range.output_gr
            sample = self.samples[out_range.sample_index]
            wlen = sample.wav_e - sample.wav_b
            autoenc_clip = self.mfcc_vc.child, self.vcs['end_grcc']
            mfcc_in = vconv.input_range(*autoenc_clip, slice_out)
            assert mfcc_in.sub_length() <= self.max_mel_len
            mfcc_add = (self.max_mel_len - mfcc_in.sub_length()) * mfcc_in.gs
            mfcc_in_pad = vconv.GridRange(mfcc_in.full, (mfcc_in.sub[0] -
                mfcc_add, mfcc_in.sub[1]), mfcc_in.gs)

            if mfcc_in_pad.valid():
                break

        preproc = self.mfcc_vc, self.mfcc_vc
        full_wav_in = vconv.GridRange((0, wlen), (0, wlen), 1)
        full_mel_in = vconv.output_range(*preproc, full_wav_in)

        enc = self.mfcc_vc.child, self.vcs['last_upsample']
        dec = self.vcs['beg_grcc'], self.vcs['end_grcc'] 

        lcond_pad = vconv.output_range(*enc, mfcc_in_pad)
        wav_in = vconv.input_range(*dec, slice_out)

        # slice of wav tensor to be input to decoder
        dec_wav_slice = vconv.tensor_slice(full_wav_in, wav_in.sub)

        # slice of internally computed local condition tensor
        lcond_slice = vconv.tensor_slice(lcond_pad, wav_in.sub)

        # slice of mel tensor to be input to encoder
        mel_in_slice = vconv.tensor_slice(full_mel_in, mfcc_in_pad.sub)

        # slice of wav buffer to be input to loss function
        loss_wav_slice = vconv.tensor_slice(wav_in, slice_out.sub)

        return SampleSlice(sample.wav_b, sample.mel_b, dec_wav_slice,
                mel_in_slice, loss_wav_slice, lcond_slice, sample.voice_index)


    def post_init(self, virtual_convs):
        """
        Initializes:
        self.vbatch
        self.slices
        Depends on information computed from the model, so must be
        called after model construction.
        """
        self.vcs = virtual_convs
        self.init_geometry()
        self.init_slices()


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
        Populates self.snd_slice, self.mel_slice, self.mask, and
        self.slice_voice_index
        Random state is from torch.{get,set}_rng_state().  It is on the CPU,
        not GPU.
        """
        vb = VirtualBatch(self.batch_size, self.max_wav_len, self.max_mel_len,
                self.n_mel_chan)
        vb.mel_input.detach_()
        vb.mel_input.requires_grad_(False)
        for b in range(vb.batch_size):
            vb.set_one(b, self.calc_slice(), self)
        vb.mel_input.requires_grad_(True)

        assert vb.valid()
        if self.target_device:
            vb.to(self.target_device)

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


