# functions for preprocessing and loading data

# parse a user-provided list of ID's and path names to .wav files
# use the list to construct the input
# perhaps it's better to abstract away the function for reading a file
# from that used to extract a unit of
import numpy as np
from sys import stderr, maxsize

#                                      => slice
# catalog -> repeat -> shuffle -> skip => slice
#                                      => slice 

# 1. There is no real need to have independent generators passing messages back.
# 2. The skipping doesn't need to actually read files, just needs to skip through
#    the sample catalog.  Though it needs to carefully interact with the shuffling
#    mechanism: in fact, if we adhere to the rule that we only shuffle the whole
#    catalog each time, then we only need to skip by mod(catalog_size)
#    But, we do need to initialize the step correctly.
from functools import total_ordering

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




class IterPosition(object):
    '''Completely describes data position.  To be used in a global save/restore
    logic for checkpointing.  '''

    def __init__(self, rand_state, data_position):

        data_position.sort()
        self.rand_state = rand_state
        self.start_epoch = data_position[0].epoch
        self.start_file_index = data_position[0].file_index
        self.slice_indices = [d.slice_index for d in data_position] 

    @classmethod
    def default(cls, batch_sz):
        rand_state = np.random.get_state()
        data_position = [Position(0, 0, 0)] * batch_sz
        return cls(rand_state, data_position)


class MaskedSliceWav(object):

    def __init__(self, sam_file, sample_rate, slice_sz, batch_sz, recep_field_sz):
        self.sam_file = sam_file
        self.sample_rate = sample_rate
        self.slice_sz = slice_sz
        self.batch_sz = batch_sz
        if self.slice_sz < recep_field_sz:
            print('Error: slice size of {} too small for receptive field size of {}'.format(
                self.slice_sz, recep_field_sz))
            raise ValueError
        self.recep_field_sz = recep_field_sz
        self.position = IterPosition.default(batch_sz)
        self.sample_catalog = []
        with open(self.sam_file) as sam_fh:
            for s in sam_fh.readlines():
                (vid, wav_path) = s.strip().split('\t')
                self.sample_catalog.append([int(vid), wav_path])

        
    def update(self, data_position):
        '''Run this in a thread together with other state updates'''
        rs = np.random.get_state()
        self.position = IterPosition(rs, data_position)



    def get_max_id(self):
        max_el, _ = max(self.sample_catalog, key=lambda x: x[0])
        return max_el


    def _wav_gen(self):
        import librosa
        epoch = self.position.start_epoch

        while True:
            shuffle_index = np.random.permutation(len(self.sample_catalog))
            shuffle_catalog = [self.sample_catalog[i] for i in shuffle_index] 
            for file_index, s in enumerate(shuffle_catalog):
                if epoch == self.position.start_epoch \
                        and file_index < self.position.start_file_index:
                    continue
                vid, wav_path = s[0], s[1]
                wav, _ = librosa.load(wav_path, self.sample_rate)
                #print('Parsing ', wav_path, file=stderr)
                yield epoch, file_index, vid, wav
            epoch += 1


    def _gen_concat_slice_factory(self, wav_gen, batch_chan):
        '''factory function for creating a new generator function.  The
        returned function encloses wav_gen.
        '''

        def slice_gen():
            '''generates a slice from concatenated set of .wav files.

            concatenate slices of self.slice_sz where:
            wav[t] = wav_val
            ids[t] = mapping_id 

            mapping_id = voice_id for valid, or zero for invalid windows
            generates (spliced_wav, spliced_ids)
            '''
            wav = np.empty(0, np.float)
            ids = np.empty(0, np.int32)
            rf_sz = self.recep_field_sz
            epoch, file_index, slice_index = 0, 0, 0 
            fast_forward = True
            lead = np.full(rf_sz - 1, -1)

            while True:
                while len(wav) < self.slice_sz:
                    try:
                        # import pdb; pdb.set_trace()
                        epoch, file_index, vid, wav_nxt = next(wav_gen) 
                        slice_index = 0

                        wav_sz = len(wav_nxt)
                        if wav_sz < rf_sz:
                            print(('Warning: skipping length {} wav file (voice id {}).  '
                                    + 'Shorter than receptive field size of {}').format( 
                                    wav_sz, vid, rf_sz))
                            continue
                        wav = np.concatenate((wav, wav_nxt), axis=0)
                        ids = np.concatenate((ids, lead, np.full(wav_sz - len(lead), vid)), axis=0)
                        assert len(wav) == len(ids)
                    except StopIteration:
                        return

                wav_slice, wav = wav[:self.slice_sz + rf_sz], wav[self.slice_sz:]
                ids_slice, ids = ids[:self.slice_sz + rf_sz], ids[self.slice_sz:]

                # Fast-forward if loaded position is 
                if fast_forward and slice_index < self.position.slice_indices[batch_chan]:
                    continue
                slice_index += 1

                yield Position(epoch, file_index, slice_index), wav_slice, ids_slice
                fast_forward = False

        return slice_gen 


    def batch_gen(self):
        '''generates a batch of concatenated slices
        yields:
        position[b] = (epoch, file_index, slice_index)
        wav[b][t] = amplitude
        ids[b][t] = vid or zero (mask)
        b = batch, t = timestep, c = channel
        '''
        wav_gen = self._wav_gen()

        # construct batch_sz slice generators, each sharing the same wav_gen
        slice_gens = [self._gen_concat_slice_factory(wav_gen, c)() for c in range(self.batch_sz)]

        while True:
            try:
                # import pdb; pdb.set_trace()
                batch = [next(g) for g in slice_gens]
                positions = [b[0] for b in batch]
                wav = np.stack([b[1] for b in batch])
                ids = np.stack([b[2] for b in batch])
                yield positions, wav, ids
            except StopIteration:
                break

