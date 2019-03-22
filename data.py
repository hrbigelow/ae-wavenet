import numpy as np
from functools import total_ordering
from sys import stderr

def _greatest_lower_bound(a, q): 
    '''return largest i such that a[i] <= q.  assume a is sorted.
    if q < a[0], return -1'''
    l, u = 0, len(a) - 1 
    while (l < u): 
        m = u - (u - l) // 2 
        if a[m] <= q: 
            l = m 
        else: 
            u = m - 1 
    return l or -1 + (a[l] <= q) 


class VirtualPermutation(object):
    # closest primes to 2^1, ..., 2^40, generated with:
    # for p in $(seq 1 40); do 
    #     i=$(echo ' 2^'$p'' | bc);
    #     primesieve 1 $i -n -q;
    # done
    primes = [3, 5, 11, 17, 37, 67, 131, 257, 521, 1031, 2053, 4099, 8209,
            16411, 32771, 65537, 131101, 262147, 524309, 1048583, 2097169,
            4194319, 8388617, 16777259, 33554467, 67108879, 134217757,
            268435459, 536870923, 1073741827, 2147483659, 4294967311,
            8589934609, 17179869209, 34359738421, 68719476767, 137438953481,
            274877906951, 549755813911, 1099511627791]


    @classmethod
    def n_items(cls, requested_n_items):
        ind = _greatest_lower_bound(cls.primes, requested_n_items)
        if ind == -1:
            raise InvalidArgument
        return cls.primes[ind]


    def __init__(self, rand_state, requested_n_items):
        self.rand_state = rand_state
        self.n_items = self.n_items(requested_n_items)

    def permutation_gen_fn(self, beg, cnt):
        '''
        # Generate cnt elements of a random permutation of [0, self.n_items) 
        # beg is the logical position *within* the virtual permutation.
        # beg can be None, which means there is no previous checkpoint.
        # cnt is the number of positions to return
        From accepted answer:
        https://math.stackexchange.com/questions/2522177/ \
                generating-random-permutations-without-storing-additional-arrays
        '''
        for n in reversed(self.primes):
            if n <= self.n_items:
                break
        assert beg <= n and cnt >= 0 and beg + cnt <= n
        a = self.rand_state.randint(0, n, 1, dtype='int64')[0]
        # We choose a range not too close to 0 or n, so that we get a
        # moderately fast moving circle.
        b = self.rand_state.randint(n//5, (4*n)//5, 1, dtype='int64')[0]
        for iter_pos in range(beg, beg + cnt):
            yield iter_pos, (a + iter_pos*b) % n


class Checkpoint(object):
    def __init__(self, rand_state, lead_wavgen_pos, perm_gen_pos):
        self.lead_wavgen_rand_state = rand_state
        self.lead_wavgen_pos = lead_wavgen_pos
        self.perm_gen_pos = perm_gen_pos
    def __str__(self):
        from pickle import dumps
        from hashlib import md5
        return 'rand: {}, lead_wavgen_pos: {}, perm_gen_pos: {}'.format(
                md5(dumps(self.lead_wavgen_rand_state)).hexdigest(),
                self.lead_wavgen_pos, 
                self.perm_gen_pos)


class WavSlices(object):
    '''
    Outline:
    1. Load the list of .wav files with their IDs into sample_catalog
    2. Generate items from sample_catalog in random order
    3. Load up to wav_buf_sz (nearest prime number lower bound) timesteps
    4. Yield them in a rotation-random order using (a + i*b) mod N technique.

    a WavSlices instance can report its full state to a client, and allows the
    client to restore its state, thus allowing checkpointing from an arbitrary
    point, including the start.  This is necessary for completely repeatable
    experiments.

    From a purely statistical ideal, this generator would break up the entire
    data set into its distinct sample windows, and then yield them in a random
    order.  However, since the encoder and decoder work in convolutional
    windows, this approach would miss the opportunity to reuse overlapping
    convolutional output values between overlapping samples.  So, we instead
    yield groups of n_win samples as one chunk.

    However, yielding these windows randomly across the data set would be
    somewhat inefficient, because it would mean re-reading an entire wav file
    each time a new sample group was needed.  Instead, we take the following
    approach:

    1.  Load a 'wav buffer' of complete wav files into memory, consuming up to
    a user-specified amount of memory.

    2.  Yield slices from that buffer in a random permutation ordering.  Only a
    fraction of all possible slices, set by frac_permutation_use, is yielded.

    3.  Once this fraction is yielded, the wav buffer is reloaded with the next
    available wav files, and the process is repeated.

    A lower value of frac_perm_use will result in more frequent reloading of
    the wav buffer, but it will also cause the yielded slices to more closely
    resemble a globally random order across the whole data set.

    Also, if the user memory (requested_wav_buf_sz) is as large as the full
    data set, then the order will be globally random, and frac_permutation_use
    should be set to 1 in order to minimize buffer reloads.
    '''

    def __init__(self, sam_file, n_win, n_batch, sample_rate,
            frac_permutation_use, requested_wav_buf_sz):
        '''
        '''
        self.sam_file = sam_file
        self.n_batch = n_batch
        self.n_win = n_win

        # Call self.set_receptive_field() after building the model
        self.recep_field_sz = None
        self.sample_rate = sample_rate
        if frac_permutation_use <= 0 or frac_permutation_use > 1.0:
            raise ValueError

        self.frac_perm_use = frac_permutation_use

        self.rand_state = np.random.mtrand.RandomState()
        self.wav_gen = None
        
        # Used in checkpointing state
        self.wavgen_rand_state = None
        self.wavgen_pos = None 
        self.lead_wavgen_rand_state = None 
        self.lead_wavgen_pos = None 
        self.perm_gen_pos = None 

        # Used in save/restore (initialized in enable_checkpointing)
        self.ckpt_dir = None
        self.ckpt_file_template = None


        # estimated number of total slices we can process in a buffer
        # of requested size (= number of time steps)
        est_n_slices = int(requested_wav_buf_sz / self.n_win)

        self.perm = VirtualPermutation(self.rand_state, est_n_slices)
        self.wav_buf = []
        self.wav_ids = []
        self.vstart = []
        self.sample_catalog = []
        with open(self.sam_file) as sam_fh:
            for s in sam_fh.readlines():
                (vid, wav_path) = s.strip().split('\t')
                self.sample_catalog.append([int(vid), wav_path])

        self.current_epoch = 1 

    def set_receptive_field(self, recep_field_sz):
        self.recep_field_sz = recep_field_sz


    def n_speakers(self):
        return len(set(id for id,__ in self.sample_catalog))

    def slice_size(self):
        return self.n_win + self.recep_field_sz - 1

    def enable_file_checkpointing(self, ckpt_dir, ckpt_file_template):
        '''prepare data module for checkpointing'''
        # Unfortunately, Python doesn't provide a way to hold an open directory
        # handle, so we just check whether the directory path exists and is
        # writable during this call.
        import os
        if not os.access(ckpt_dir, os.R_OK|os.W_OK):
            raise ValueError('Cannot read and write checkpoint directory {}'.format(ckpt_dir))
        # test if ckpt_file_template is valid  
        try:
            test_file = ckpt_file_template.format(1000)
        except IndexError:
            test_file = ''
        # '1000' is 2 longer than '{}'
        if len(test_file) != len(ckpt_file_template) + 2:
            raise ValueError('Checkpoint template "{}" ill-formed. ' 
                    '(should have exactly one "{{}}")'.format(ckpt_file_template))
        try:
            test_path = '{}/{}'.format(ckpt_dir, test_file)
            if not os.access(test_path, os.R_OK):
                fp = open(test_path, 'w')
                fp.close()
                os.remove(fp.name)
        except IOError:
            raise ValueError('Cannot create a test checkpoint file {}'.format(test_path))

        self.ckpt_dir = ckpt_dir
        self.ckpt_file_template = ckpt_file_template
        

    def state_to_ckpt(self):
        if self.lead_wavgen_rand_state is None:
            print('No generator created yet, so no checkpoint defined.')
            return None
        return Checkpoint(self.lead_wavgen_rand_state,
                self.lead_wavgen_pos, self.perm_gen_pos)


    def ckpt_to_state(self, ckpt):
        '''After calling restore_checkpoint, a call to _slice_gen_fn produces a
        generator object with the same state as when it was saved'''
        self.rand_state.set_state(ckpt.lead_wavgen_rand_state)
        self.wav_gen = self._wav_gen_fn(ckpt.lead_wavgen_pos)
        self.lead_wavgen_pos = ckpt.lead_wavgen_pos
        self.perm_gen_pos = ckpt.perm_gen_pos

    def ckpt_to_file(self, ckpt, step):
        pass

    def file_to_ckpt(self, step):
        pass
    

    def _wav_gen_fn(self, pos):
        '''random order generation of one epoch of whole wav files.'''
        import librosa
        self.wavgen_rand_state = self.rand_state.get_state()
        self.wavgen_pos = pos

        def gen_fn():

            shuffle_index = self.rand_state.permutation(len(self.sample_catalog))
            shuffle_catalog = [self.sample_catalog[i] for i in shuffle_index] 
            for iter_pos, s in enumerate(shuffle_catalog[pos:], pos):
                vid, wav_path = s[0], s[1]
                wav, _ = librosa.load(wav_path, self.sample_rate)
                # print('Parsing ', wav_path, file=stderr)
                self.wavgen_pos = iter_pos
                yield iter_pos, vid, wav

            # Completed a full epoch
            self.current_epoch += 1

        return gen_fn()


    def _load_wav_buffer(self):
        '''Fully load the wav file buffer, which is a list of full wav arrays
        taking up up to the requested memory size.
        
        Consumes remaining contents of current wav_gen, reissuing generators as
        needed.  This function relies on the checkpoint state through
        self.wav_gen and self.rand_state
        '''
        vpos = 0
        self.wav_buf = []
        self.wav_ids = []
        self.vstart = []

        if self.wav_gen is None:
            self.wav_gen = self._wav_gen_fn(0)

        self.lead_wavgen_rand_state = self.wavgen_rand_state
        self.lead_wavgen_pos = self.wavgen_pos

        self.offset = self.rand_state.randint(0, self.n_win, 1, dtype='int32')[0] 
        last_v_start = self.offset + (self.perm.n_items - 1) * self.n_win
        while vpos < last_v_start:
            try:
                iter_pos, vid, wav = next(self.wav_gen)
            except StopIteration:
                self.wav_gen = self._wav_gen_fn(0)
                iter_pos, vid, wav = next(self.wav_gen)

            self.wav_buf.append(wav)
            self.wav_ids.append(vid)
            self.vstart.append(vpos)
            vpos += len(wav) - self.slice_size()


    def _slice_gen_fn(self):
        '''Extracts slices from the wav buffer (see self._load_wav_buffer) in 
        a random permutation ordering.  Only extracts a fraction of the total
        available slices (set by self.frac_perm_use) before exhausting.
        of the 

        '''
        self._load_wav_buffer()
        if self.perm_gen_pos is None:
            self.perm_gen_pos = 0

        def gen_fn():
            perm_gen = self.perm.permutation_gen_fn(self.perm_gen_pos,
                    int(self.perm.n_items * self.frac_perm_use))
            for iter_pos, vind in perm_gen:
                vpos = self.offset + vind * self.n_win
                wav_file_ind = _greatest_lower_bound(self.vstart, vpos)
                wav_off = vpos - self.vstart[wav_file_ind]

                # self.perm_gen_pos gives the position that will be yielded next
                self.perm_gen_pos = iter_pos + 1
                yield wav_file_ind, wav_off, vind, \
                        self.wav_ids[wav_file_ind], \
                        self.wav_buf[wav_file_ind][wav_off:wav_off + self.slice_size()]

            # We've exhausted the iterator, next position should be zero
            self.perm_gen_pos = 0
        return gen_fn()


    def batch_slice_gen_fn(self):
        '''infinite generator for batched slices of wav files'''

        def gen_fn(sg):
            b = 0
            wavs = np.empty((self.n_batch, self.slice_size()), dtype='float64')
            ids = np.empty(self.n_batch, dtype='int32')
            while True:
                while b < self.n_batch:
                    try:
                        wav_file_ind, wav_off, vind, wav_id, wav_slice = next(sg)
                    except StopIteration:
                        sg = self._slice_gen_fn()
                        continue
                    wavs[b,:] = wav_slice
                    ids[b] = wav_id
                    b += 1
                yield ids, wavs
                b = 0

        return gen_fn(self._slice_gen_fn())






            


