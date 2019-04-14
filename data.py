import numpy as np
from sys import stderr
import util
import librosa
# If you get 'NoBackendError' exception, try
# sudo apt-get install libav-tools

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
    def compute_n_items(cls, requested_n_items):
        ind = util.greatest_lower_bound(cls.primes, requested_n_items)
        if ind == -1:
            raise InvalidArgument
        return cls.primes[ind]

    def __init__(self, rand_state, requested_n_items):
        self.rand_state = rand_state
        self.n_items = self.compute_n_items(requested_n_items)

    def permutation_gen_fn(self, beg, end):
        '''
        # Generate (end - beg) elements of a random permutation of [0, self.n_items) 
        # beg is the logical start position within the virtual permutation.
        # end is one after the last position of the virtual permutation. 
        From accepted answer:
        https://math.stackexchange.com/questions/2522177/ \
                generating-random-permutations-without-storing-additional-arrays
        '''
        for n in reversed(self.primes):
            if n <= self.n_items:
                break
        if not (0 <= beg <= end):
            raise RuntimeError('beg must be between 0 and n ({}), got '
                    '{}'.format(end, beg))
        if not (0 < end <= n):
            raise RuntimeError('end must be between 1 and {}'.format(n))

        a = self.rand_state.randint(0, n, 1, dtype='int64')[0]
        # We choose a range not too close to 0 or n, so that we get a
        # moderately fast moving circle.
        b = self.rand_state.randint(n//5, (4*n)//5, 1, dtype='int64')[0]
        for iter_pos in range(beg, end):
            yield iter_pos, (a + iter_pos*b) % n


class WavSlices(object):
    '''
    Outline:
    1. Load the list of .wav files with their IDs into sample_catalog
    2. Generate items from sample_catalog in random order
    3. Load up to wav_buf_sz (nearest prime number lower bound) timesteps
    4. Yield them in a rotation-random order using (a + i*b) mod N technique.

    WavSlices instances are pickle-able through __getstate__ and __setstate__.

    From a purely statistical ideal, this generator would break up the entire
    data set into its distinct sample windows, and then yield them in a random
    order.  However, since the encoder and decoder work in convolutional
    windows, this approach would miss the opportunity to reuse overlapping
    convolutional output values between overlapping samples.  So, we instead
    yield groups of n_sample_win samples as one chunk.

    However, yielding these windows randomly across the data set would be
    somewhat inefficient, because it would mean re-reading an entire wav file
    each time a new sample group was needed.  Instead, we take the following
    approach:

    1.  Load a 'wav buffer' of complete wav files into memory, consuming up to
    a user-specified amount of memory.

    2.  Yield slices from that buffer in a random permutation ordering.  Only a
    fraction of all possible slices, set by frac_perm_use, is yielded.

    3.  Once this fraction is yielded, the wav buffer is reloaded with the next
    available wav files, and the process is repeated.

    A lower value of frac_perm_use will result in more frequent reloading of
    the wav buffer, but it will also cause the yielded slices to more closely
    resemble a globally random order across the whole data set.

    Also, if the user memory (requested_wav_buf_sz) is as large as the full
    data set, then the order will be globally random, and frac_perm_use
    should be set to 1 in order to minimize buffer reloads.
    '''

    def __init__(self, sample_catalog, sample_rate, frac_perm_use,
            req_wav_buf_sz):
        state = { 
                'sample_catalog': sample_catalog,
                'sample_rate': sample_rate,
                'frac_perm_use': frac_perm_use,
                'req_wav_buf_sz': req_wav_buf_sz
                }
        self._initialize(state)

    def _initialize(self, state):
        '''Shared initialization code for __init__ and __setstate__'''
        # Used in checkpointing state
        self.wavgen_rand_state = None
        self.wavgen_pos = None 
        self.lead_wavgen_rand_state = None 
        self.lead_wavgen_pos = None 
        self.perm_gen_pos = None 
        self.current_epoch = 1 
        self.rand_state = np.random.mtrand.RandomState()
        
        brs_key = 'buffer_rand_state'
        self.__dict__.update({ k:v for k,v in state.items() if k != brs_key})
        rs = state.get(brs_key)
        if rs is not None:
            self.rand_state.set_state(rs)

        if not (0 <= self.frac_perm_use <= 1.0):
            raise ValueError('frac_perm_use must be between 0 and 1.  got '
                    '{}'.format(self.frac_perm_use))

        self.wav_gen = None
        self.wav_buf = []
        self.wav_ids = []
        self.vstart = []
        self.speaker_id_map = dict((v,k) for k,v in enumerate(self.speaker_ids()))

    def speaker_ids(self):
        return set(id for id,__ in self.sample_catalog)

    def num_speakers(self):
        return len(self.speaker_ids())

    def set_geometry(self, n_batch, slice_size, n_sam_per_slice):
        self.n_batch = n_batch
        self.n_sample_win = n_sam_per_slice
        self.slice_size = slice_size 
        est_n_slices = int(self.req_wav_buf_sz / self.n_sample_win)
        self.perm = VirtualPermutation(self.rand_state, est_n_slices)

    def __getstate__(self):
        '''Return the state of this data generator.  Analogous to
        torch.nn.Module.state_dict()'''
        if self.buffer_rand_state is None:
            raise RuntimeError('No generator created yet, so no checkpoint defined.')
        state = {
                'buffer_rand_state': self.buffer_rand_state,
                'buffer_pos': self.buffer_pos,
                'perm_gen_pos': self.perm_gen_pos,
                'sample_catalog': self.sample_catalog,
                'sample_rate': self.sample_rate,
                'frac_perm_use': self.frac_perm_use,
                'req_wav_buf_sz': self.req_wav_buf_sz,
                'current_epoch': self.current_epoch
                }
        return state

    def __setstate__(self, state):
        '''update the generator state with that checkpointed.'''
        self._initialize(state)

    def _wav_gen_fn(self, pos):
        '''random order generation of one epoch of whole wav files.'''
        # State for save/restore
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
            print('Data: finished epoch {}'.format(self.current_epoch), file=stderr)
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

        # State information for save/restore.
        self.buffer_rand_state = self.wavgen_rand_state
        self.buffer_pos = self.wavgen_pos

        self.offset = self.rand_state.randint(0, self.n_sample_win, 1, dtype='int32')[0] 
        last_v_start = self.offset + (self.perm.n_items - 1) * self.n_sample_win
        while vpos < last_v_start:
            try:
                iter_pos, vid, wav = next(self.wav_gen)
            except StopIteration:
                self.wav_gen = self._wav_gen_fn(0)
                iter_pos, vid, wav = next(self.wav_gen)

            self.wav_buf.append(wav)
            self.wav_ids.append(vid)
            self.vstart.append(vpos)
            vpos += len(wav) - self.slice_size
        print('Data: loaded wav buffer', file=stderr)

    def _slice_gen_fn(self):
        '''Extracts slices from the wav buffer (see self._load_wav_buffer) in 
        a random permutation ordering.  Only extracts a fraction of the total
        available slices (set by self.frac_perm_use) before exhausting.
        '''
        # State for save/restore: self.perm_gen_pos
        self._load_wav_buffer()
        if self.perm_gen_pos is None:
            self.perm_gen_pos = 0

        def gen_fn():
            # random state for self.perm is determined from
            # self.wav_buffer_rand_state, so don't need to store it.
            perm_gen = self.perm.permutation_gen_fn(self.perm_gen_pos,
                    int(self.perm.n_items * self.frac_perm_use))
            for iter_pos, vind in perm_gen:
                vpos = self.offset + vind * self.n_sample_win
                wav_file_ind = util.greatest_lower_bound(self.vstart, vpos)
                wav_off = vpos - self.vstart[wav_file_ind]

                # self.perm_gen_pos gives the position that will be yielded next
                self.perm_gen_pos = iter_pos + 1
                yield wav_file_ind, wav_off, vind, \
                        self.wav_ids[wav_file_ind], \
                        self.wav_buf[wav_file_ind][wav_off:wav_off + self.slice_size]

            # We've exhausted the iterator, next position should be zero
            self.perm_gen_pos = 0
        return gen_fn()

    def batch_slice_gen_fn(self):
        '''infinite generator for batched slices of wav files'''
        def gen_fn(sg):
            b = 0
            wavs = np.empty((self.n_batch, self.slice_size), dtype='float32')
            ids = np.empty(self.n_batch, dtype='int32')
            inds = np.empty(self.n_batch, dtype='long')
            while True:
                while b < self.n_batch:
                    try:
                        wav_file_ind, wav_off, vind, wav_id, wav_slice = next(sg)
                    except StopIteration:
                        sg = self._slice_gen_fn()
                        continue
                    wavs[b,:] = wav_slice
                    ids[b] = wav_id
                    inds[b] = self.speaker_id_map[wav_id] 
                    b += 1
                yield ids, inds, wavs
                b = 0

        return gen_fn(self._slice_gen_fn())


def parse_sample_catalog(sam_file):
    try:
        sample_catalog = []
        with open(sam_file) as sam_fh:
            for s in sam_fh.readlines():
                (vid, wav_path) = s.strip().split('\t')
                sample_catalog.append([int(vid), wav_path])
    except (FileNotFoundError, IOError):
        raise RuntimeError("Couldn't open or read samples file {}".format(sam_file))
    return sample_catalog

