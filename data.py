import numpy as np
from functools import total_ordering

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

    def permutation_gen_fn(self, pos):
        '''
        # needed to generate pseudo-random permutations with constant memory
        From accepted answer:
        https://math.stackexchange.com/questions/2522177/ \
                generating-random-permutations-without-storing-additional-arrays
        '''
        for n in reversed(self.primes):
            if n <= self.n_items:
                break
        assert pos <= n
        a = self.rand_state.randint(0, n, 1, dtype='int64')[0]
        b = self.rand_state.randint(n/2, n, 1, dtype='int64')[0]
        for i in range(pos, n):
            yield i, (a + i*b) % n
        return


@total_ordering
class Position(object):
    def __init__(self, epoch, file_index, slice_index):
        self.epoch = epoch
        self.file_index = file_index
        self.slice_index = slice_index

    def __lt__(self, other):
        return self.epoch < other.epoch or \
                self.epoch == other.epoch and \
                self.file_index < other.file_index

    def __eq__(self, other):
        return self.epoch == other.epoch and \
                self.file_index == other.file_index


class Checkpoint(object):
    def __init__(self, rand_state, lead_wavgen_pos, perm_gen_pos):
        self.rand_state = rand_state
        self.lead_wavgen_pos = lead_wavgen_pos
        self.perm_gen_pos = perm_gen_pos


class WavSlices(object):
    '''
    Outline:
    1. Load the list of .wav files with their IDs into sample_catalog
    2. Generate items from sample_catalog in random order
    3. Load up to wav_buf_sz (nearest prime number lower bound) timesteps
    4. Yield them in a rotation-random order using (a + i*b) mod N technique.

    WavSlices allows a client to read its (WavSlices) full state, and to
    restore its state.  One can thus save/restore checkpoint at an arbitrary
    point, including the start, thus allowing for completely repeatable
    experiments.
    '''

    def __init__(self, sam_file, slice_sz, batch_sz, sample_rate,
            requested_wav_buf_sz):
        self.sam_file = sam_file
        self.slice_sz = slice_sz
        self.batch_sz = batch_sz
        self.sample_rate = sample_rate
        self.rand_state = np.random.mtrand.RandomState()
        #self.position = IterPosition()
        self.wav_gen = None
        self.perm = VirtualPermutation(self.rand_state, requested_wav_buf_sz)
        self.ckpt = Checkpoint(None, 0, 0)
        try:
            self.wav_buf = np.empty(self.perm.n_items)
        except MemoryError:
            from sys import stderr, exit
            print('Out of memory.  Please use smaller requested_wav_buf_sz.', file=stderr)
            exit(1) 

        self.sample_catalog = []
        with open(self.sam_file) as sam_fh:
            for s in sam_fh.readlines():
                (vid, wav_path) = s.strip().split('\t')
                self.sample_catalog.append([int(vid), wav_path])

        # Note: need save/restore logic here
        # 
        self.position = None


    def restore(self):
        '''After calling restore, a call to _slice_gen_fn produces a generator
        object with the same state as when it was saved'''
        
        self.rand_state.set_state(self.ckpt.prev_wg_rand_state)
        self.wav_gen = self.wav_gen_fn(self.ckpt.lead_wg_pos)
        self.pg_pos = self.ckpt.pg_pos
    
    def save(self):
        self.ckpt.lead_wg_pos = 0 # !!!
        self.ckpt.pg_pos = 0 # !!!
        assert False


    def _wav_gen_fn(self, pos):
        '''random order generation of one epoch of whole wav files.'''
        import librosa

        self.ckpt.prev_wg_rand_state = self.rand_state.get_state()
        shuffle_index = self.rand_state.permutation(len(self.sample_catalog))
        shuffle_catalog = [self.sample_catalog[i] for i in shuffle_index] 
        for i, s in enumerate(shuffle_catalog[pos:], pos):
            vid, wav_path = s[0], s[1]
            wav, _ = librosa.load(wav_path, self.sample_rate)
            #print('Parsing ', wav_path, file=stderr)
            yield i, vid, wav

        # Completed a full epoch
        self.position.epoch += 1


    def _load_wav_buffer(self):
        '''Fully load the wav file buffer.  Consumes remaining contents of
        current wav_gen, reissuing generators as needed.'''
        tpos, vpos, ind = 0, 0, 0
        self.wav_starts = []
        self.offsets = []
        if self.wav_gen is None:
            self.wav_gen = self._wav_gen_fn(0)

        while tpos < self.perm.n_items:
            try:
                i, vid, wav = next(self.wav_gen)
            except StopIteration:
                self.wav_gen = self._wav_gen_fn(0)
                i, vid, wav = next(self.wav_gen)

            n_add = min(len(wav), self.perm.n_items - tpos)
            self.wav_buf[tpos:tpos + n_add] = wav[0:n_add]
            self.wav_starts.append(tpos)
            self.offsets.append(vpos)
            self.ids.append(vid)
            tpos += n_add 
            vpos += n_add - self.slice_sz # !!! check this
        self.wg_pos = i


    def _slice_gen_fn(self):
        self._load_wav_buffer()
        perm_gen = self.perm.permutation_gen_fn(self.pg_pos)
        self.pg_pos = 0
        for slice_num, vpos in perm_gen:
            i = _greatest_lower_bound(self.voff, vpos)
            off = vpos - self.voff[i]
            wpos = self.wav_starts[i] + off
            yield slice_num, vpos, self.ids[i], self.wav_buf[wpos:wpos + self.slice_sz]


    def batch_slice_gen_fn(self):
        pass


