# functions for preprocessing and loading data

# parse a user-provided list of ID's and path names to .wav files
# use the list to construct the input
# perhaps it's better to abstract away the function for reading a file
# from that used to extract a unit of

# decision
# 1. parse the main file into a map
# 2. enqueue the files
# 3. load them as needed into empty slots. 

# fuctions needed

import numpy as np
import ckpt
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



class MaskedSliceWav(object):

    def __init__(self, sess, sam_file, sample_rate, slice_sz, prefetch_sz,
            batch_sz, n_keep_checkpoints, ckpt_path, resume_step):
        self.sam_file = sam_file
        self.sample_rate = sample_rate
        self.prefetch_sz = prefetch_sz
        self.slice_sz = slice_sz
        self.batch_sz = batch_sz
        self.random_seed = np.random.randint(maxsize)
        self.ckpt_position = 0
        
    # OK
    def init_sample_catalog(self):
        self.sample_catalog = []
        with open(self.sam_file) as sam_fh:
            for s in sam_fh.readlines():
                (vid, wav_path) = s.strip().split('\t')
                self.sample_catalog.append([int(vid), wav_path])


    # OK
    def set_receptive_field_size(self, r_sz):
        if self.slice_sz < r_sz:
            print('Error: slice size of {} too small for receptive field size of {}'.format(
                self.slice_sz, r_sz))
            raise ValueError
        self.recep_field_sz = r_sz

    # OK
    def get_max_id(self):
        max_el, _ = max(self.sample_catalog, key=lambda x: x[0])
        return max_el

    # OK
    def _gen_path(self):
        '''load a sample file with the format:
        voice_id /path/to/wav.flac
        voice_id /path/to/wav.flac
        ...
        generate tuples (voice_id, wav_path)
        '''
        for s in self.sample_catalog:
            vid, wav_path = s[0], s[1]
            #print('Parsing ', wav_path, file=stderr)
            yield vid, wav_path


    # OK
    def _wav_gen(self, path_gen):
        '''consume an iterator that yields [voice_id, wav_path].
        load the .wav file contents into a vector and return a tuple
        generate tuples (voice_id, [wav_val, wav_val, ...])
        '''
        datum_count = self.ckpt_position
        while True:
            try:
                datum_count += 1
                vid, wav_path = next(path_gen)
                wav, _ = librosa.load(wav_path, self.sample_rate)
                #print('loaded wav and mel of size {}'.format(wav.data.nbytes + mel.data.nbytes),
                #        file=stderr)
                yield datum_count, int(vid), wav
            except StopIteration: 
                break
        return


    def _gen_concat_slice_factory(self, wav_gen):
        '''factory function for creating a new generator function.  The
        returned function encloses wav_gen.
        '''

        def gen_fcn():
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

            while True:
                while len(wav) < self.slice_sz:
                    try:
                        # import pdb; pdb.set_trace()
                        datum_count, vid, wav_nxt = next(wav_gen) 
                        wav_sz = len(wav_nxt)
                        if wav_sz < rf_sz:
                            print(('Warning: skipping length {} wav file (voice id {}).  '
                                    + 'Shorter than receptive field size of {}').format( 
                                    wav_sz, vid, rf_sz))
                            continue
                        wav.append(wav_nxt)
                        ### !!! check that contents and length of ids is correct
                        ids = np.append(np.full(-1, rf_sz - 1),
                                np.full(vid, len(wav_nxt) - rf_sz + 1))
                        assert len(wav) == len(ids)
                    except StopIteration:
                        return

                wav_slice, wav = wav[:self.slice_sz + rf_sz], wav[self.slice_sz:]
                ids_slice, ids = ids[:self.slice_sz + rf_sz], ids[self.slice_sz:]
                yield wav_slice, ids_slice
                return

        return gen_fcn


    def _gen_slice_batch(self, path_gen):
        '''generates a batch of concatenated slices
        yields:
        wav[b][t] = amplitude
        mel[b][t][c] = freq
        ids[b][t] = vid or zero (mask)
        b = batch, t = timestep, c = channel
        '''
        wav_gen = self._wav_gen(path_gen)

        # construct batch_sz slice generators, each sharing the same wav_gen
        gens = [self._gen_concat_slice_factory(wav_gen)() for _ in range(self.batch_sz)]

        while True:
            try:
                # this is probably expensive
                # import pdb; pdb.set_trace()
                batch = [next(g) for g in gens]
                # cnt represents how many wav file readings have been made
                # (repeated readings of the same wav file are counted separately)
                latest_file_read_count = batch[-1][0]
                wav = np.stack([b[1] for b in batch])
                ids = np.stack([b[2] for b in batch])
                yield latest_file_read_count, wav, ids
            except StopIteration:
                # this will be raised if wav_itr runs out
                break


    def build(self):
        '''parse a sample file and create a ts.data.Dataset of concatenated,
        labeled slices from it.
        call this to create a fresh dataset.
            '''
        zero_d = tf.TensorShape([])
        two_d = tf.TensorShape([self.batch_sz, None])
        three_d = tf.TensorShape([self.batch_sz, None, self.mel_spectrum_sz])

        with tf.name_scope('dataset'):
            with tf.name_scope('sample_map'):
                ds = tf.data.Dataset.from_generator(
                        self._gen_path,
                        (tf.int32, tf.string, tf.string),
                        (zero_d, zero_d, zero_d))

            with tf.name_scope('shuffle_repeat'):
                ds = ds.repeat()
                buf_sz = len(self.sample_catalog)
                ds = ds.shuffle(buffer_size=buf_sz, seed=self.random_seed)
                ds = ds.skip(self.ckpt_position)
                # this iterator must be initializable because it is dependent
                # on self.ckpt_position and self.random_seed, which must
                # be initialized
                self.path_itr = ds.make_initializable_iterator()

                # used this so a reassignment of 'itr' doesn't break the code
                def gen_wrap():
                    return self._gen_slice_batch(gen_wrap.itr)
                gen_wrap.itr = self.path_itr 

            with tf.name_scope('slice_batch'):
                ds = tf.data.Dataset.from_generator(
                        gen_wrap,
                        (tf.int64, tf.int32, tf.float32, tf.int32),
                        (zero_d, two_d, three_d, two_d))

            with tf.name_scope('prefetch'):
                ds = ds.prefetch(buffer_size=self.prefetch_sz)

        self.dataset_itr = ds.make_initializable_iterator()

        # these two determine where in the dataset we will resume
        self.add_saveable_objects({
            'random_seed': self.random_seed,
            'ckpt_position': self.ckpt_position
            })

        self.add_initializable_ops([self.path_itr, self.dataset_itr])

    def save(self, step, read_count):
        op = tf.assign(self.ckpt_position, read_count)
        if tf.executing_eagerly():
            pass
        else:
            self.sess.run(op)
        return super().save(step)


