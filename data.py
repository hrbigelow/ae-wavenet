# Preprocess Data
from sys import stderr
import pickle
import librosa
import numpy as np
import torch
from torch import nn
import vconv
import copy

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


class SampleGeom(object):
    """
    Information about each Sample File 
    """
    def __init__(self, out_beg, out_end, n_slices,
            snd_len, mel_len, snd_voff, mel_voff):
        self.out_beg = out_beg # timestep offset in the input of first output prediction 
        self.out_end = out_end # timestep offset in the input of last output prediction
        self.n_slices = n_slices # number of windowed batches in this sample
        self.snd_len = snd_len # number of timesteps in this sample 
        self.mel_len = mel_len # number of mfcc elements
        self.snd_voff = snd_voff # cumulative values of snd_len
        self.mel_voff = mel_voff # cumulative values of mel_len
        
    def __repr__(self):
        return '(o: [{}, {}), ns: {}, sl: {}, ml: {}, soff: {}, moff: {})\n'.format(
                self.out_beg, self.out_end, self.n_slices, self.snd_len,
                self.mel_len, self.snd_voff, self.mel_voff
                )


class Slice(nn.Module):
    def __init__(self, ind_pfx, batch_size):
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

        # index contains arrays voice_index[], n_snd_elem[], n_mel_elem[], wav_path[]
        self.__dict__.update(index)
        self.n_files = len(self.n_snd_elem)

        self.mfcc_vc = vconv.VirtualConv(
                filter_info=self.window_size,
                stride=self.hop_size,
                parent=None, name='MFCC'
        )


    def num_speakers(self):
        return len(set(self.voice_index))

    def post_init(self, model, index_file_prefix, n_sam_per_slice_requested,
            max_gpu_mem_bytes):
        """
        Inputs:
        Initialize snd_voffset, mel_voffset
        Initialize n_total_win_batch and end_vc values
        Initialize vcs for setting the geometry of slices
        """
        self.slice_voff = np.empty(self.n_files, dtype=np.int32)
        self.sample = []

        # last vconv in the autoencoder
        self.wave_beg_vc = model.decoder.vc['beg_grcc']
        self.wave_end_vc = model.decoder.vc['end_grcc']

        vc_rng = self.mfcc_vc, self.wave_end_vc
        req_out_e = n_sam_per_slice_requested

        # Calculate actual window_batch_size from requested
        # The encoder downsamples wave input by a factor of 320.
        # WaveNet upsamples the conditioning vectors by a factor of 320.
        
        __, in_e, in_l = vconv.recep_field(vc_rng[0], vc_rng[1], 0, req_out_e,
                100000)
        __, out_e, out_l = vconv.output_range(vc_rng[0], vc_rng[1], 0, in_e,
                100000)

        self.window_batch_size = out_e

        geom = SampleGeom(0, 0, 0, 0, 0, 0, 0)
        slice_voff = 0
        self.n_win_batch = 0
        for i, (snd_len, mel_len) in enumerate(zip(self.n_snd_elem, self.n_mel_elem)):
            geom.out_beg, geom.out_end, __ = vconv.output_range(
                self.wave_beg_vc, self.wave_end_vc, 0, snd_len, snd_len)
            assert geom.out_beg == 0
            geom.n_slices = geom.out_end // self.window_batch_size
            geom.snd_len = snd_len
            geom.mel_len = mel_len
            self.sample.append(copy.copy(geom))
            geom.snd_voff += geom.snd_len
            geom.mel_voff += geom.mel_len
            self.slice_voff[i] = slice_voff 
            slice_voff += geom.n_slices
            # we don't know the window batch size for mel, but it is guaranteed
            # to have the same number of slices due to the model geometry

        self.n_win_batch = slice_voff

        total_snd_elem = sum(map(lambda g: g.snd_len, self.sample))
        total_mel_elem = sum(map(lambda g: g.mel_len, self.sample))

        dat_file = index_file_prefix + '.dat'
        mel_file = index_file_prefix + '.mel'

        if self.snd_dtype is np.uint8:
            buf = torch.ByteStorage.from_file(dat_file, shared=False, size=total_snd_elem)
            snd_data = torch.ByteTensor(buf)
        elif self.snd_dtype is np.uint16:
            buf = torch.ShortStorage.from_file(dat_file, shared=False, size=total_snd_elem)
            snd_data = torch.ShortTensor(buf)
        elif self.snd_dtype is np.int32:
            buf = torch.IntStorage.from_file(dat_file, shared=False, size=total_snd_elem)
            snd_data = torch.IntTensor(buf)

        mel_buf = torch.FloatStorage.from_file(mel_file, shared=False, size=total_mel_elem)
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
            self.snd_data = snd_data
            self.mel_data = mel_data

        self.register_buffer('in_snd_slice', torch.empty((self.batch_size, 0),
            dtype=self.snd_data.dtype))
        self.register_buffer('out_snd_slice', torch.empty((self.batch_size, 0),
            dtype=self.snd_data.dtype))
        self.register_buffer('in_mel_slice', torch.empty((self.batch_size, 0),
            dtype=torch.float))
        self.register_buffer('slice_voice_index', torch.empty((self.batch_size, 0),
            dtype=torch.int))


    def next_slice(self):
        """
        Get a random slice of a file, together with its start position and ID.
        Populates self.snd_slice, self.mel_slice, self.mask, and
        self.slice_voice_index
        """
        picks = np.random.random_integers(0, self.n_win_batch, self.batch_size)
        for b, slice_voff in enumerate(picks):
            file_i = util.greatest_lower_bound(self.slice_voff, slice_voff)
            slice_i = slice_voff - self.slice_voff[file_i]

            # Calculate the required input ranges for snd and mel to generate
            # the picked sample window
            # [sam_b, sam_e) is the range of the desired output
            sam_b = slice_i * self.window_batch_size
            sam_e = sam_b + self.window_batch_size
            sam_l = self.n_snd_elem[file_i]

            in_mel_range = vconv.recep_field(
                    self.mfcc_vc, self.wave_end_vc, sam_b, sam_e, sam_l
            )
            in_snd_range = vconv.recep_field(
                    self.wave_beg_vc, self.wave_end_vc, sam_b, sam_e, sam_l
            )
            # In the following transformation:
            # wav --[MFCC]-> mel --[Encoder]-> embeddings --[WaveNet Upsampling] -> local_cond
            # the local_cond vector is one-per-timestep, like the wav input
            # Due to the window-based geometry of [MFCC], [Encoder], and
            # [WaveNet Upsampling], the elements of local_cond correspond to wav.
            # This calculates the sub-range in wav corresponding to local_cond.
            in_snd_shadow = vconv.shadow(
                    self.wave_beg_vc, self.wave_end_vc, 
                    in_snd_range[0], in_snd_range[1], in_snd_range[2] 
                    )
             
            mel_voff = self.sample[file_i].mel_voff
            snd_voff = self.sample[file_i].snd_voff

            self.in_mel_slice[b] = self.mel_data[(mel_voff +
                in_mel_range[0]):(mel_voff + in_mel_range[1])]
            self.in_snd_slice[b] = self.snd_data[(snd_voff +
                in_snd_range[0]):(snd_voff + in_snd_range[1])]
            self.out_snd_slice[b] = self.snd_data[(snd_voff + sam_b):(snd_voff
                + sam_e)]
            self.slice_voice_index[b] = self.voice_index[file_i]

