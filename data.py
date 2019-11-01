# Preprocess Data
from sys import stderr
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

def convert(catalog, pfx, n_quant, sample_rate=16000, win_sz=400, hop_sz=160,
        n_mels=80, n_mfcc=13):
    """
    Convert all input data and save in a pack of .dat, .ind, and .mel files.
    The .mel file contains the concatenated contents of computed mfcc coefficients.
    The .dat file contains the concatenated contents of the mu-encoded wav files.
    The .ind file contains the indexing information.
    """
    mfcc_proc = mfcc.ProcessWav(sample_rate, win_sz, hop_sz, n_mels, n_mfcc)

    if n_quant <= 2**8:
        snd_dtype = np.uint8
    elif n_quant <= 2**15:
        snd_dtype = np.int16
    else:
        snd_dtype = np.int32

    snd_file = pfx + '.dat'
    ind_file = pfx + '.ind'
    mel_file = pfx + '.mel'
    ind = { 'voice_index': [], 'n_snd_elem': [], 'n_mel_elem': [], 'snd_path': [] }
    n_snd_elem = 0
    n_mel_elem = 0
    n_mel_chan = None
    speaker_ids = set(id for id,__ in catalog)
    speaker_id_map = dict((v,k) for k,v in enumerate(speaker_ids))

    with open(snd_file, 'wb') as snd_fh, open(mel_file, 'wb') as mel_fh:
        for (voice_id, snd_path) in catalog:
            snd, _ = librosa.load(snd_path, sample_rate)
            snd_mu = util.mu_encode_np(snd, n_quant).astype(snd_dtype)
            # mel: C, T  (n_mels, n_timesteps)
            # reshape to T, C and flatten
            mel = mfcc_proc.func(snd)
            if n_mel_chan is None:
                n_mel_chan = mel.shape[0]

            mel = mel.transpose((1, 0)).flatten()
            snd_fh.write(snd_mu.data)
            mel_fh.write(mel.data)
            ind['voice_index'].append(speaker_id_map[voice_id])
            ind['n_snd_elem'].append(snd.size)
            ind['n_mel_elem'].append(mel.size)
            ind['snd_path'].append(snd_path)
            if len(ind['voice_index']) % 100 == 0:
                print('Converted {} files of {}.'.format(len(ind['voice_index']),
                    len(catalog), file=stderr))
                stderr.flush()
            n_snd_elem += snd.size
            n_mel_elem += mel.size

    with open(ind_file, 'wb') as ind_fh:
        index = { 
                'window_size': win_sz,
                'hop_size': hop_sz,
                'n_snd_elem': n_snd_elem,
                'n_mel_elem': n_mel_elem,
                'n_mel_chan': n_mel_chan,
                'snd_dtype': snd_dtype,
                'n_quant': n_quant
                }
        index.update(ind)
        pickle.dump(index, ind_fh)



# Represents a slice of a sample plus the geometry
SampleSlice = namedtuple('SampleSlice', [
    'wav_offset',    # Offset into full wav dataset
    'mel_offset',    # Offset into full mel dataset
    'dec_wav_slice', # wav slice input to decoder
    'mel_in_slice',  # mel slice input to encoder
    'loss_wav_slice',# slice of wav decoder input for input to loss function 
    'lcond_slice'    # slice of lcond tensor to pass on to the decoder
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


class VirtualBatch(object):
    def __init__(self, batch_size, max_wav_len, max_mel_len):
        self.batch_size = batch_size
        self.voice_index = torch.empty(batch_size, dtype=torch.long)
        self.lcond_slice = [None] * batch_size 
        self.loss_wav_slice = [None] * batch_size 
        self.wav_input = torch.empty(batch_size, max_wav_len)
        self.mel_input = torch.empty(batch_size, max_mel_len) 

    def add(self, b, sample_slice, wav_tensor_slice, mel_tensor_slice):
        self.voice_index[b] = sample_slice.voice_index
        self.lcond_slice[b] = sample_slice.lcond_slice
        self.loss_wav_slice[b] = sample_slice.loss_wav_slice
        self.wav_input[b] = wav_tensor_slice
        self.mel_input[b] = mel_tensor_slice

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
    def __init__(self, ind_pfx, batch_size, window_batch_size):
        super(Slice, self).__init__()
        try:
            ind_file = ind_pfx + '.ind'
            with open(ind_file, 'rb') as ind_fh:
                idx = pickle.load(ind_fh)
        except IOError:
            print('Could not open index file {}.'.format(ind_file), file=stderr)
            stderr.flush()
            exit(1)

        self.batch_size = batch_size
        self.window_batch_size = window_batch_size
        self.samples = []
        woff, moff = 0, 0
        for vi, wlen, mlen, path in zip(
                idx['voice_index'], idx['n_snd_elem'], idx['n_mel_elem'],
                idx['snd_path']):
            self.samples.append(SpokenSample(
                voice_index=vi,
                wav_b=woff,
                wav_e=woff + wlen,
                mel_b=moff,
                mel_e=moff + mlen,
                file_path=path
                ))
            woff += wlen
            moff += mlen

        self.window_size = idx['window_size']
        self.hop_size = idx['hop_size']
        self.n_mel_chan = idx['n_mel_chan']
        self.snd_dtype = idx['snd_dtype']
        self.n_quant = idx['n_quant']

        self.mfcc_vc = vconv.VirtualConv(
                filter_info=self.window_size,
                stride=self.hop_size,
                parent=None, name='MFCC'
        )


    def num_speakers(self):
        return len(set(map(lambda s: s.voice_index, self.samples)))

    def post_init(self, model, index_file_prefix, max_gpu_mem_bytes):
        """
        """
        # last vconv in the autoencoder
        self.wave_beg_vc = model.decoder.vc['beg_grcc']
        self.wave_end_vc = model.decoder.vc['end_grcc']

        autoenc = self.mfcc_vc, self.wave_end_vc
        autoenc_clip = self.mfcc_vc.child, self.wave_end_vc
        enc = self.mfcc_vc.child, self.last_upsample_vc
        dec = self.wave_beg_vc, self.wave_end_vc
        mfcc = self.mfcc_vc, self.mfcc_vc

        # Calculate max length of mfcc encoder input and wav decoder input
        w = self.window_batch_size
        max_spacing = vconv.max_spacing(*autoenc, 1)
        max_mfcc_len = 0
        max_wav_len = 0
        for b in range(max_spacing):
            out = vconv.GridRange((0, 10000), (b, b + w), 1)
            mfcc = vconv.input_range(*autoenc_clip, out)
            max_mfcc_len = max(mfcc.sub_length(), max_mfcc_len)
            wav = vconv.input_range(*dec, out)
            max_wav_len = max(wav.sub_length(), max_wav_len)

        self.max_mel_input_length = max_mfcc_len
        self.max_wav_input_length = max_wav_len

        # generate all slices
        self.slices = []
        for file_i, sam in enumerate(self.samples): 
            # all windows right aligned, fitting in this wav data
            wlen = sam.wav_e - sam.wav_b
            mlen = sam.mel_e - sam.mel_b
            full_wav_in = vconv.GridRange((0, wlen), (0, wlen), 1)
            full_mel_in = vconv.GridRange((0, wlen), (0, wlen), 1)
            full_out = vconv.output_range(*autoenc, full_wav_in) 
            assert full_out.gs == 1
            slice_out = vconv.GridRange(full_out.full, 
                    (full_out.sub[1] - w, full_out.sub[1], full_out.gs))

            while True:
                mfcc_in = vconv.input_range(*autoenc_clip, slice_out)
                mfcc_in_pad = mfcc_in
                mfcc_add = max_mfcc_len - mfcc_in.sub_length()
                mfcc_in_pad.sub[0] -= mfcc_in.gs * mfcc_add

                wav_in = vconv.input_range(*dec, slice_out)
                wav_in_pad = wav_in
                wav_add = max_wav_len - wav_in.sub_length()
                wav_in_pad.sub[0] -= wav_in.gs * wav_add
                if not (mfcc_in_pad.valid() and wav_in_pad.valid()):
                    break

                # slice of wav tensor to be input to decoder
                dec_wav_slice = vconv.tensor_slice(full_wav_in, wav_in_pad.sub)

                # slice of mel tensor to be input to encoder
                mel_in_slice = vconv.tensor_slice(full_mel_in, mfcc_in_pad.sub)

                # slice of wav buffer to be input to loss function
                loss_wav_slice = vconv.tensor_slice(full_wav_in, wav_out.sub)

                lcond_pad = vconv.output_range(*enc, mfcc_in_pad)
                # slice of internally computed local condition tensor
                # (output by encoder+upsampling) to pass on to decoder
                lcond_slice = vconv.tensor_slice(lcond_pad, wav_in_pad.sub)

                self.slices.append(
                        SampleSlice(sam.wav_b, sam.mel_b, dec_wav_slice,
                            mel_in_slice, loss_wav_slice, lcond_slice)
                        )

        # Load dataset into memory
        snd_length = self.samples[-1].wav_e
        mel_length = self.samples[-1].mel_e 

        dat_file = index_file_prefix + '.dat'
        mel_file = index_file_prefix + '.mel'

        if self.snd_dtype is np.uint8:
            buf = torch.ByteStorage.from_file(dat_file, shared=False, size=snd_length)
            snd_data = torch.ByteTensor(buf)
        elif self.snd_dtype is np.uint16:
            buf = torch.ShortStorage.from_file(dat_file, shared=False, size=snd_length)
            snd_data = torch.ShortTensor(buf)
        elif self.snd_dtype is np.int32:
            buf = torch.IntStorage.from_file(dat_file, shared=False, size=snd_length)
            snd_data = torch.IntTensor(buf)

        mel_buf = torch.FloatStorage.from_file(mel_file, shared=False, size=mel_length)
        mel_data = torch.FloatTensor(mel_buf).reshape((-1, self.n_mel_chan))

        # Store entire dataset in GPU memory if possible 
        total_bytes = (
                snd_data.nelement() * snd_data.element_size() +
                mel_data.nelement() * mel_data.element_size()
                )
        self.gpu_resident = (total_bytes <= max_gpu_mem_bytes)
        if self.gpu_resident:
            self.register_buffer('snd_data', snd_data)
            self.register_buffer('mel_data', mel_data)
        else:
            # These still need to be properties, but we won't register them
            self.snd_data = snd_data
            self.mel_data = mel_data


    def next_slice(self):
        """
        Get a random slice of a file, together with its start position and ID.
        Populates self.snd_slice, self.mel_slice, self.mask, and
        self.slice_voice_index
        """
        picks = np.random.random_integers(0, len(self.slices), self.batch_size)
        vb = VirtualBatch(self.batch_size, self.max_wav_input_length,
                self.max_mel_input_length)

        for b, s_i in enumerate(picks):
            sl = self.slices[s_i]
            wo = sl.wav_offset
            mo = sl.mel_offset
            ws = sl.dec_wav_slice
            ms = sl.mel_in_slice
            wav_in = self.snd_data[wo + ws[0]:wo + ws[1]]
            mel_in = self.mel_data[mo + ms[0]:mo + ms[1]]
            vb.add(b, sl, wav_in, mel_in)

        assert vb.valid()
        return vb

