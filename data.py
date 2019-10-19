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

class VirualOffsets(object):
    def __init__(self, cumul_snd_offset, cumul_mel_offset, cumul_num_slices):
        self.cumul_snd_offset = cumul_snd_offset
        self.cumul_mel_offset = cumul_mel_offset
        self.cumul_vslices = cumul_num_slices


class Slice(nn.Module):
    def __init__(self, ind_pfx, max_gpu_mem_bytes, batch_size, window_batch_size):
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
        self.window_batch_size = window_batch_size

        # index contains arrays voice_index[], n_snd_elem[], n_mel_elem[], wav_path[]
        self.__dict__.update(index)
        self.n_files = len(self.n_snd_elem)
        self.snd_offset = np.empty(self.n_files, dtype=np.int32)
        self.mel_offset = np.empty(self.n_files, dtype=np.int32)

        self.mfcc_vc = vconv.VirtualConv(
                filter_info=self.window_size,
                stride=self.hop_sz,
                parent=None, name='MFCC'
        )

        snd_off = 0
        for i in range(self.n_files):
            self.snd_offset[i] = snd_off
            snd_off += self.n_snd_elem[i]
        total_snd_elem = snd_off

        mel_off = 0
        for i in range(self.n_files):
            self.mel_offset[i] = mel_off
            mel_off += self.n_mel_elem[i]
        total_mel_elem = mel_off

        dat_file = ind_pfx + '.dat'
        mel_file = ind_pfx + '.mel'

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
            self.snd_data = data
            self.mel_data = mel_data

        self.register_buffer('snd_slice', torch.empty((batch_size, 0),
            dtype=self.snd_data.dtype))
        self.register_buffer('snd_slice_out', torch.empty((batch_size, 0),
            dtype=self.snd_data.dtype))
        self.register_buffer('mel_slice', torch.empty((batch_size, 0),
            dtype=torch.float))
        self.register_buffer('voice_index', torch.empty((batch_size, 0),
            dtype=torch.int))

    def num_speakers(self):
        return len(set(self.voice_index))

    def post_init(self, model, n_sam_per_slice_requested):
        """
        Inputs:
        Initialize snd_voffset, mel_voffset
        Initialize n_total_win_batch and end_vc values
        Initialize vcs for setting the geometry of slices
        """
        self.voffset = np.empty((self.n_files), dtype=np.int32)
        self.n_sam_win = np.empty((self.n_files), dtype=np.int32)

        # last vconv in the autoencoder
        self.pre_upsample_vc = model.decoder.pre_upsample_vc
        self.last_upsample_vc = model.decoder.last_upsample_vc
        self.last_grcc_vc = model.decoder.last_grcc_vc

        # Calculate actual window_batch_size from requested
        in_b, in_e = vconv.rfield(self.pre_upsample_vc, self.last_grcc_vc,
                0, n_sam_per_slice_requested, n_sam_per_slice_requested)
        assert in_b == 0
        out_b, out_e = vconv.ifield(self.pre_upsample_vc, self.last_grcc_vc,
                0, in_e, in_e)
        assert out_b == 0
        self.window_batch_size = out_e
        self.n_total_win_batch = 0

        stat = VirtualOffsets(0, 0, 0)
        for i, (snd_len, mel_len) in enumerate(zip(self.n_snd_elem, self.n_mel_elem)):
            self.virtual_offsets[i] = stat
            n_sam_win = snd_len // self.window_batch_size
            # we don't know the window batch size for mel, but it is guaranteed
            # to have the same number of slices due to the model geometry
            stat.cumul_snd_len += snd_len
            stat.cumul_mel_len += mel_len
            stat.cumul_num_slices += n_sam_win
            self.n_total_win_batch += n_sam_win 


    def next_slice(self):
        """
        Get a random slice of a file, together with its start position and ID.
        Populates self.snd_slice, self.mel_slice, self.mask, and
        self.slice_voice_index
        """
        picks = np.random(0, self.n_total_win_batch, self.batch_size)
        for vwin, b in enumerate(picks):
            file_i = util.greatest_lower_bound(self.snd_voffset, vwin)
            win_i = vwin - self.voffset[file_i]

            # Calculate the required input ranges for snd and mel to generate
            # the picked sample window
            # [sam_b, sam_e) is the range of the desired output
            sam_b = win_i * self.window_batch_size
            sam_e = sam_b + self.window_batch_size
            sam_l = self.n_sam_windows[file_i] * self.window_batch_size
            mel_in_b, mel_in_e = vconv.rfield(
                    self.mfcc_vc.child, self.last_grcc_vc, sam_b, sam_e, sam_l
            )
            dec_in_b, dec_in_e = vconv.rfield(
                    self.mfcc_vc, self.last_grcc_vc, sam_b, sam_e, sam_l
            )
            # In the following transformation:
            # wav --[MFCC]-> mel --[Encoder]-> embeddings --[WaveNet Upsampling] -> local_cond
            # the local_cond vector is one-per-timestep, like the wav input
            # Due to the window-based geometry of [MFCC], [Encoder], and
            # [WaveNet Upsampling], the elements of local_cond correspond to wav.
            # This calculates the sub-range in wav corresponding to local_cond.
            # 
            guide_sam_b, guide_sam_e = vconv.shadow(
                    self.mfcc_vc, self.last_upsample_vc, 
                    sam_b, sam_e, sam_e
            )

            snd_off = self.snd_offset[file_i]
            mel_off = self.mel_offset[file_i]
            self.snd_dec_input[b] = self.snd_data[(snd_off + dec_in_b):(snd_off + dec_in_e)]
            self.snd_slice_out[b] = self.snd_data[(snd_off + guide_sam_b):(snd_off + guide_sam_e)]
            self.mel_slice[b] = self.mel_data[(mel_off + mel_in_b):(mel_off + mel_in_e)]
            self.slice_voice_index[b] = self.voice_index[file_i]

