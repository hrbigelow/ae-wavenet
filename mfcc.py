# Functions for extracting Mel and MFCC information from a raw Wave file

# T = time step, F = frame (every ~100 timesteps or so)
# Waveform: shape(T), values (limited range integers)
# MFCC + d + a: shape(F, 13 * 3 see figure 1)
# Output of Layer 1 Conv: shape(F, 39, 768)

# From paper:
# 80 log-mel filterbank features extracted every 10ms from 25ms-long windows
# 13 MFCC features

# From librosa.feature.melspectrogram:
# n_fft (# timesteps in FFT window)
# hop_length (# timesteps between successive window positions)

# From librosa.filters.mel:
# n_fft ("# FFT components" (is this passed on to melspectrogram?)
# n_mels (# mel bands to generate)

# From librosa.feature.mfcc):
# n_mfcc (# of MFCCs to return)

import torch
import numpy as np
import vconv 
import math

class ProcessWav(object):
    def __init__(self, sample_rate=16000, win_sz=400, hop_sz=160, n_mels=80,
            n_mfcc=13, name=None):
        self.sample_rate = sample_rate
        self.window_sz = win_sz
        self.hop_sz = hop_sz
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_out = n_mfcc * 3
        self.vc = vconv.VirtualConv(filter_info=self.window_sz, stride=self.hop_sz,
                parent=None, name=name)

    def func(self, wav):
        import librosa
        # See padding_notes.txt 
        # NOTE: This function can't be executed on GPU due to the use of
        # librosa.feature.mfcc
        # C, T: n_mels, n_timesteps
        # Output: C, T
        # This assert doesn't seem to work when we just want to process an entire wav file
        adj = 1 if self.window_sz % 2 == 0 else 0
        adj_l_wing_sz = self.vc.l_wing_sz + adj 

        left_pad = adj_l_wing_sz % self.hop_sz
        trim_left = adj_l_wing_sz // self.hop_sz
        trim_right = self.vc.r_wing_sz // self.hop_sz

        # wav = wav.numpy()
        wav_pad = np.concatenate((np.zeros(left_pad), wav), axis=0) 
        mfcc = librosa.feature.mfcc(y=wav_pad, sr=self.sample_rate,
                n_fft=self.window_sz, hop_length=self.hop_sz,
                n_mels=self.n_mels, n_mfcc=self.n_mfcc)

        def mfcc_pred_output_size(in_sz, window_sz, hop_sz):
            '''Reverse-engineered output size calculation derived by observing the
            behavior of librosa.feature.mfcc'''
            n_extra = 1 if window_sz % 2 == 0 else 0
            n_pos = in_sz + n_extra
            return n_pos // hop_sz + (1 if n_pos % hop_sz > 0 else 0)

        assert mfcc.shape[1] == mfcc_pred_output_size(wav_pad.shape[0],
            self.window_sz, self.hop_sz)

        mfcc_trim = mfcc[:,trim_left:-trim_right or None]

        mfcc_delta = librosa.feature.delta(mfcc_trim)
        mfcc_delta2 = librosa.feature.delta(mfcc_trim, order=2)
        mfcc_and_derivatives = np.concatenate((mfcc_trim, mfcc_delta, mfcc_delta2), axis=0)

        return torch.tensor(mfcc_and_derivatives)

