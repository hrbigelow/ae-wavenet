# Preprocess Data
from sys import stderr, exit
import pickle
import librosa
import numpy as np
import torch
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


class VirtualBatch(nn.Module):
    def __init__(self, batch_size, max_wav_len, max_mel_len, mel_chan):
        super(VirtualBatch, self).__init__()
        self.batch_size = batch_size
        self.n_mel_chan = mel_chan
        self.register_buffer('voice_index', torch.empty(batch_size,
            dtype=torch.long))
        self.lcond_slice = [None] * batch_size 
        self.loss_wav_slice = [None] * batch_size 
        self.register_buffer('wav_input', torch.empty(batch_size, max_wav_len))
        self.register_buffer('mel_input', torch.empty(batch_size, mel_chan,
            max_mel_len)) 

    def __repr__(self):
        fmt = ('voice_index: {}\nlcond_slice: {}\nloss_wav_slice: {}\n' +
                'wav_input.shape: {}\nmel_input.shape: {}\n')
        return fmt.format(self.voice_index, self.lcond_slice,
                self.loss_wav_slice, self.wav_input.shape,
                self.mel_input.shape)

    def set(self, b, sample_slice, data_source):
        ss = sample_slice
        self.voice_index[b] = ss.voice_index
        wo = ss.wav_offset
        mo = ss.mel_offset
        dws = ss.dec_wav_slice
        mis = ss.mel_in_slice

        self.lcond_slice[b] = ss.lcond_slice 
        self.loss_wav_slice[b] = ss.loss_wav_slice 
        self.wav_input[b] = data_source.snd_data[wo + dws[0]:wo + dws[1]] 
        self.mel_input[b] = data_source.mel_data[mo + mis[0]:mo +
                mis[1],:].transpose(1, 0)


    def valid(self):
        lc_len = self.lcond_len()
        lw_len = self.loss_wav_len()
        return (
                all(map(lambda lw: lw[1] - lw[0] == lw_len,
                    self.loss_wav_slice)) and 
                all(map(lambda lc: lc[1] - lc[0] == lc_len,
                    self.lcond_slice))
                )

    def lcond_len(self):
        return self.lcond_slice[0][1] - self.lcond_slice[0][0]

    def loss_wav_len(self):
        return self.loss_wav_slice[0][1] - self.loss_wav_slice[0][0]


class Slice(nn.Module):
    """
    Defines the current batch of data
    """
    def __init__(self, dat_file, slice_file, batch_size, window_batch_size,
            gpu_resident):
        self.init_args = {
                'dat_file': dat_file,
                'slice_file': slice_file,
                'batch_size': batch_size,
                'window_batch_size': window_batch_size,
                'gpu_resident': gpu_resident
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
        self.dat_file = self.init_args['dat_file']
        self.slice_file = self.init_args['slice_file']
        self.batch_size = self.init_args['batch_size']
        self.window_batch_size = self.init_args['window_batch_size']
        self.gpu_resident = self.init_args['gpu_resident']

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
                dat['snd_dtype'], self.n_mel_chan, self.gpu_resident)

        self.mfcc_vc = vconv.VirtualConv(
                filter_info=mfcc_pars['window_size'], stride=mfcc_pars['hop_size'],
                parent=None, name='MFCC'
        )

    def __setstate__(self, init_args):
        self.init_args = init_args 
        self._initialize()
        self._initialize_vbatch()
        self._initialize_slices()


    def __getstate__(self):
        return self.init_args


    def num_speakers(self):
        return len(set(map(lambda s: s.voice_index, self.samples)))


    def init_vbatch(self, virtual_convs):
        """
        Initializes:
        self.vbatch
        """
        # Calculate max length of mfcc encoder input and wav decoder input
        w = self.window_batch_size
        beg_grcc_vc = virtual_convs['beg_grcc']
        end_grcc_vc = virtual_convs['end_grcc']
        autoenc = self.mfcc_vc, end_grcc_vc
        autoenc_clip = self.mfcc_vc.child, end_grcc_vc
        dec = beg_grcc_vc, end_grcc_vc 

        max_spacing = vconv.max_spacing(*autoenc, 1)
        max_mel_in_len = 0
        for b in range(max_spacing):
            out = vconv.GridRange((0, 100000), (b, b + w), 1)
            mfcc = vconv.input_range(*autoenc_clip, out)
            # print(mfcc.sub_length(), end=' ')
            max_mel_in_len = max(mfcc.sub_length(), max_mel_in_len)

        # Calculate decoder wav input length
        slice_out = vconv.GridRange((0, 100000), (0, w), 1)
        max_wav_in_len = vconv.input_range(*dec, slice_out).sub_length()
        
        print('Max Mel input length: {}'.format(max_mel_in_len), file=stderr)
        stderr.flush()
        self.init_args['max_wav_in_len'] = max_wav_in_len
        self.init_args['max_mel_in_len'] = max_mel_in_len
        self._initialize_vbatch()


    def _initialize_vbatch(self):
        max_wav_in_len = self.init_args['max_wav_in_len']
        max_mel_in_len = self.init_args['max_mel_in_len']
        n_mel_chan = self.mel_data.shape[1]
        self.vbatch = VirtualBatch(self.batch_size, max_wav_in_len,
                max_mel_in_len, n_mel_chan)


    def init_slices(self, virtual_convs, slice_file):
        """
        Initialize:
        self.slices
        """

        # generate all slices
        win_size = self.window_batch_size
        max_mel_len = self.vbatch.mel_input.shape[2]
        self.slices = []
        for sam in self.samples: 
            print('Processing {}'.format(sam.file_path), file=stderr)
            stderr.flush()
            self._add_slices(sam, virtual_convs, max_mel_len, win_size)

        self.init_args['slice_file'] = slice_file
        with open(slice_file, 'wb') as slice_fh: 
            pickle.dump(self.slices, slice_fh)

    def _initialize_slices(self):
        slice_file = self.init_args['slice_file']
        with open(slice_file, 'rb') as slice_fh:
            self.slices = pickle.load(slice_fh)

        

    def _add_slices(self, sample, vcs, max_mel_len, win_size):
        """
        Add slices for this sample
        """
        preproc = self.mfcc_vc, self.mfcc_vc
        enc = self.mfcc_vc.child, vcs['last_upsample']
        dec = vcs['beg_grcc'], vcs['end_grcc'] 
        autoenc = self.mfcc_vc, vcs['end_grcc']
        autoenc_clip = self.mfcc_vc.child, vcs['end_grcc']

        wlen = sample.wav_e - sample.wav_b
        full_wav_in = vconv.GridRange((0, wlen), (0, wlen), 1)
        full_mel_in = vconv.output_range(*preproc, full_wav_in)
        full_out = vconv.output_range(*autoenc, full_wav_in) 
        assert full_out.gs == 1
        slice_out = vconv.GridRange(full_out.full, 
                (full_out.sub[1] - win_size, full_out.sub[1]), full_out.gs)

        while True:
            # We should instead derive the padding some other way, so that
            # the mfcc_in_pad and wav_in_pad are consistent with each other
            mfcc_in = vconv.input_range(*autoenc_clip, slice_out)
            assert mfcc_in.sub_length() <= max_mel_len
            mfcc_add = (max_mel_len - mfcc_in.sub_length()) * mfcc_in.gs
            mfcc_in_pad = vconv.GridRange(mfcc_in.full, (mfcc_in.sub[0] -
                mfcc_add, mfcc_in.sub[1]), mfcc_in.gs)

            if not mfcc_in_pad.valid():
                break

            lcond_pad = vconv.output_range(*enc, mfcc_in_pad)
            wav_in = vconv.input_range(*dec, slice_out)
            #print(wav_in)

            # slice of wav tensor to be input to decoder
            dec_wav_slice = vconv.tensor_slice(full_wav_in, wav_in.sub)

            # slice of internally computed local condition tensor
            lcond_slice = vconv.tensor_slice(lcond_pad, wav_in.sub)

            # slice of mel tensor to be input to encoder
            mel_in_slice = vconv.tensor_slice(full_mel_in, mfcc_in_pad.sub)

            # slice of wav buffer to be input to loss function
            loss_wav_slice = vconv.tensor_slice(wav_in, slice_out.sub)

            self.slices.append(
                    SampleSlice(sample.wav_b, sample.mel_b, dec_wav_slice,
                        mel_in_slice, loss_wav_slice, lcond_slice,
                        sample.voice_index)
                    )
            slice_out.sub[0] -= win_size
            slice_out.sub[1] -= win_size


    def post_init(self, virtual_convs):
        """
        Initializes:
        self.vbatch
        self.slices
        Depends on information computed from the model, so must be
        called after model construction.
        """
        self.init_vbatch(virtual_convs)
        self.init_slices(virtual_convs, self.slice_file)


    def _load_sample_data(self, snd_np, mel_np, snd_dtype, n_mel_chan,
            gpu_resident):
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

        # Store entire dataset in GPU memory if possible 
        total_bytes = (
                snd_data.nelement() * snd_data.element_size() +
                mel_data.nelement() * mel_data.element_size()
                )
        if gpu_resident:
            self.register_buffer('snd_data', snd_data)
            self.register_buffer('mel_data', mel_data)
        else:
            # These still need to be properties, but we won't register them
            self.snd_data = snd_data
            self.mel_data = mel_data



    def next_batch(self):
        """
        Get a random slice of a file, together with its start position and ID.
        Populates self.snd_slice, self.mel_slice, self.mask, and
        self.slice_voice_index
        """
        picks = np.random.random_integers(0, len(self.slices) - 1, self.batch_size)

        for b, s_i in enumerate(picks):
            self.vbatch.set(b, self.slices[s_i], self)

        assert self.vbatch.valid()
        return self.vbatch

