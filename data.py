# Preprocess Data
from sys import stderr
import pickle
import librosa
import numpy as np
import torch
from torch import nn

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
    ind = { 'voice_id': [], 'n_snd_elem': [], 'n_mel_elem': [], 'snd_path': [] }
    n_snd_elem = 0
    n_mel_elem = 0
    n_mel_chan = None

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
            ind['voice_id'].append(voice_id)
            ind['n_snd_elem'].append(snd.size)
            ind['n_mel_elem'].append(mel.size)
            ind['snd_path'].append(snd_path)
            if len(ind['voice_id']) % 100 == 0:
                print('Converted {} files of {}.'.format(len(ind['voice_id']),
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

class Slice(nn.Module):
    def __init__(self, ind_pfx, max_gpu_mem_bytes, batch_size):
        super(Slice, self).__init__()
        try:
            ind_file = ind_pfx + '.ind'
            with open(ind_file, 'rb') as ind_fh:
                index = pickle.load(ind_fh)
        except IOError:
            print('Could not open index file {}.'.format(ind_file), file=stderr)
            stderr.flush()
            exit(1)

        self.batch_size = batch_size

        # index contains arrays voice_id[], n_snd_elem[], n_mel_elem[], wav_path[]
        self.__dict__.update(index)
        self.n_files = len(self.n_snd_elem)
        self.snd_offset = np.empty(self.n_files, dtype=np.int32)
        self.mel_offset = np.empty(self.n_files, dtype=np.int32)

        snd_off = 0
        for i in range(self.n_files):
            self.snd_offset[i] = snd_off
            snd_off += self.n_snd_elem[i]
        self.total_snd_elem = snd_off

        mel_off = 0
        for i in range(self.n_files):
            self.mel_offset[i] = mel_off
            mel_off += self.n_mel_elem[i]
        self.total_mel_elem = mel_off

        dat_file = ind_pfx + '.dat'
        mel_file = ind_pfx + '.mel'

        if self.snd_dtype is np.uint8:
            buf = torch.ByteStorage.from_file(dat_file, shared=False, size=self.total_snd_elem)
            snd_data = torch.ByteTensor(buf)
        elif self.snd_dtype is np.uint16:
            buf = torch.ShortStorage.from_file(dat_file, shared=False, size=self.total_snd_elem)
            snd_data = torch.ShortTensor(buf)
        elif self.snd_dtype is np.int32:
            buf = torch.IntStorage.from_file(dat_file, shared=False, size=self.total_snd_elem)
            snd_data = torch.IntTensor(buf)

        mel_buf = torch.FloatStorage.from_file(mel_file, shared=False, size=self.total_mel_elem)
        mel_data = torch.FloatTensor(mel_buf).reshape((-1, self.n_mel_chan))

        # flag to indicate if we can directly use a GPU resident buffer
        total_bytes = snd_data.nelement() * snd_data.element_size() + \
                mel_data.nelement() * mel_data.element_size()
        self.gpu_resident = (total_bytes <= max_gpu_mem_bytes)
        if self.gpu_resident:
            self.register_buffer('snd_data', snd_data)
            self.register_buffer('mel_data', mel_data)
        else:
            # These still need to be properties, but we won't register them
            self.data = data
            self.mel_data = mel_data

        self.register_buffer('snd_slice', torch.empty((batch_size, 0),
            dtype=self.snd_data.dtype))
        self.register_buffer('mel_slice', torch.empty((batch_size, 0),
            dtype=torch.float))
        self.register_buffer('mask', torch.empty((batch_size, 0),
            dtype=torch.float))

    def num_speakers(self):
        return len(set(self.voice_id))

    def init_geometry(self, ae_wav_in, ae_mel_in, dec_wav_in, dec_out):
        self.ae_wav_in = ae_wav_in
        self.ae_mel_in = ae_mel_in
        self.dec_wav_in = dec_wav_in
        self.dec_out = dec_out
        self.voffset = np.empty((self.n_files), dtype=np.int32)
        self.n_samples = np.empty((self.n_files), dtype=np.int32)
        voff = 0
        for i, wav_len in enumerate(self.n_snd_elem):
            __, out_e = rf.get_ifield(self.ae_wav_in, self.dec_out, 0, wav_len, True)
            self.voffset[i] = voff
            self.n_samples[i] = out_e
            voff += out_e
        self.n_total_samples = voff

    def next_slice(self):
        """Get a random slice of a file, together with its start position
        and ID.  Populates self.snd_slice, self.mel_slice, and self.mask"""
        picks = np.random(0, self.n_total_samples, self.batch_size)
        for vpos, b in enumerate(picks):
            file_i = util.greatest_lower_bound(self.voffset, vpos)
            last_in = self.n_snd_elem[file_i] - 1
            last_out = self.n_samples[file_i] - 1
            sam_i = vpos - self.voffset[file_i]
            mel_in_b, mel_in_e = rf.get_rfield(self.mel_in, self.dec_out,
                    sam_i, sam_i, last_out)
            dec_in_b, dec_in_e = rf.get_rfield(self.dec_in, self.dec_out,
                    sam_i, sam_i, last_out)
            out_b, out_e = rf.get_ifield(self.ae_wav_in, self.dec_out,
                    snd_in_b, snd_in_e, last_in)

            snd_off = self.snd_offset[file_i]
            mel_off = self.mel_offset[file_i]
            self.snd_slice[b] = self.snd_data[snd_off + dec_in_b:snd_off + dec_in_e + 1]
            self.mel_slice[b] = self.mel_data[mel_off + mel_in_b:mel_off + mel_in_e + 1]
            self.mask[b].zero_()
            self.mask[b,sam_i-out_b] = 1
            assert self.mask.size()[1] == out_e - out_b

