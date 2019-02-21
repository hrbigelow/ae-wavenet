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

class MaskedSliceWav(object):

    def __init__(self, sess, sam_file, sample_rate, slice_sz, prefetch_sz,
            mel_spectrum_sz, mel_hop_sz, batch_sz, n_keep_checkpoints,
            ckpt_path, resume_step):
        super().__init__() # TODO: Checkpointing somehow
        self.sam_file = sam_file
        self.sample_rate = sample_rate
        self.prefetch_sz = prefetch_sz
        #self.mel_spectrum_sz = mel_spectrum_sz
        #self.mel_hop_sz = mel_hop_sz
        if slice_sz % mel_hop_sz != 0:
            requested_slice_sz = slice_sz
            slice_sz += mel_hop_sz - (slice_sz % mel_hop_sz)
            print('Warning: aligning slice size from {} to {} for mel_hop_sz {}'.format(
                requested_slice_sz, slice_sz, mel_hop_sz), file=stderr) 
        self.slice_sz = slice_sz
        self.batch_sz = batch_sz
        self.random_seed = np.random.randint(maxsize)
        self.ckpt_position = 0
        
    def init_sample_catalog(self):
        self.sample_catalog = []
        with open(self.sam_file) as sam_fh:
            for s in sam_fh.readlines():
                (vid, wav_path) = s.strip().split('\t')
                self.sample_catalog.append([int(vid), wav_path])

    def set_receptive_field_size(self, r_sz):
        self.recep_field_sz = r_sz

    def get_max_id(self):
        max_el, _ = max(self.sample_catalog, key=lambda x: x[0])
        return max_el

    def _gen_path(self):
        '''load a sample file with the format:
        voice_id /path/to/wav.flac
        voice_id /path/to/wav.flac
        ...
        generate tuples (voice_id, wav_path)
        '''
        def enc(s): return bytes(s, 'utf-8')
        for s in self.sample_catalog:
            vid, wav_path= s[0], s[1]
            #print('Parsing ', wav_path, file=stderr)
            yield vid, enc(wav_path), enc(mel_path)
        return


    def _wav_gen(self, path_itr):
        '''consume an iterator that yields [voice_id, wav_path].
        load the .wav file contents into a vector and return a tuple
        generate tuples (voice_id, [wav_val, wav_val, ...])
        '''
        assert not tf.executing_eagerly()
        next_el = path_itr.get_next()
        datum_count = self.sess.run(self.ckpt_position)
        while True:
            try:
                datum_count += 1
                vid, wav_path, mel_path = self.sess.run(next_el)
                wav = np.load(wav_path.decode())
                mel = np.load(mel_path.decode())
                #print('loaded wav and mel of size {}'.format(wav.data.nbytes + mel.data.nbytes),
                #        file=stderr)
                yield datum_count, int(vid), wav, mel
            except tf.errors.OutOfRangeError:
                break
        return


    def _gen_concat_slice_factory(self, wav_gen):
        '''factory function for creating a new generator
        wav_gen: generates next (vid, wav, mel) pair
        '''

        def gen_fcn():
            '''generates a slice from a virtually concatenated set of .wav files.
            consume itr, which yields [voice_id, wav_data]

            concatenate slices of self.slice_sz where:
            spliced_wav[t] = wav_val
            spliced_mel[t] = mel_spectrum_array
            spliced_ids[t] = mapping_id 

            mapping_id corresponds to voice_id for valid positions, or zero for invalid
            (positions corresponding to junction-spanning receptive field windows)

            generates (spliced_wav, spliced_ids, idmap)
            '''
            need_sz = self.slice_sz 
            spliced_wav = np.empty(0, np.float)
            spliced_mel = np.empty([0, self.mel_spectrum_sz], np.float)
            spliced_ids = np.empty(0, np.int32)
            recep_bound = self.recep_field_sz - 1

            def mc(val): return val // self.mel_hop_sz

            while True:
                try:
                    # import pdb; pdb.set_trace()
                    datum_count, vid, wav, mel = next(wav_gen) 
                    snip = len(wav) % self.mel_hop_sz
                    wav = wav[:-snip or None]
                    if len(wav) != len(mel) * self.mel_hop_sz:
                        print('Error: len(wav) = {}, len(mel) * mel_hop_sz = {}'.format(
                            len(wav), len(mel) * self.mel_hop_sz))

                except StopIteration:
                    break
                wav_sz = wav.shape[0] 
                if wav_sz < self.recep_field_sz:
                    print(('Warning: skipping length {} wav file (voice id {}).  '
                            + 'Shorter than receptive field size of {}').format( 
                            wav_sz, vid, self.recep_field_sz))
                    continue
                
                ids = np.concatenate([
                    np.full(recep_bound, 0, np.int32),
                    np.full(wav_sz - recep_bound, vid, np.int32) 
                    ])

                cur_pos = 0
                
                while need_sz <= (wav_sz - cur_pos):
                    # print(str(need_sz) + ', ' + str(wav_sz - cur_pos))
                    # use up a chunk of the current item and yield the slice
                    spliced_wav = np.append(spliced_wav,
                            wav[cur_pos:cur_pos + need_sz],
                            axis=0)

                    spliced_mel = np.append(spliced_mel,
                            mel[mc(cur_pos):mc(cur_pos + need_sz)],
                            axis=0)

                    spliced_ids = np.append(spliced_ids,
                            ids[cur_pos:cur_pos + need_sz],
                            axis=0)
                    cur_pos += need_sz 
                    yield datum_count, spliced_wav, spliced_mel, spliced_ids
                    spliced_wav = np.empty(0, np.float) 
                    spliced_mel = np.empty([0, self.mel_spectrum_sz], np.float)
                    spliced_ids = np.empty(0, np.int32)
                    need_sz = self.slice_sz 

                if cur_pos != wav_sz:
                    # append this piece of wav to the current slice 
                    spliced_wav = np.append(spliced_wav, wav[cur_pos:], axis=0)
                    spliced_mel = np.append(spliced_mel, mel[mc(cur_pos):], axis=0)
                    spliced_ids = np.append(spliced_ids, ids[cur_pos:], axis=0)
                    need_sz -= (wav_sz - cur_pos)
            return
        return gen_fcn


    def _gen_slice_batch(self, path_itr):
        '''generates a batch of concatenated slices
        yields:
        wav[b][t] = amplitude
        mel[b][t][c] = freq
        ids[b][t] = vid or zero (mask)
        b = batch, t = timestep, c = channel
        '''
        # !!! this is where the single global item counter needs to be.
        # iter_cnt how many times wav_gen has been iterated.
        # construct the single (iter_cnt, vid, wav, mel) generator
        if tf.executing_eagerly():
            wav_gen = self._wav_gen_eager(path_itr)
        else:
            wav_gen = self._wav_gen(path_itr)

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
                mel = np.stack([b[2] for b in batch])
                ids = np.stack([b[3] for b in batch])
                yield latest_file_read_count, wav, mel, ids
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


    def get_itr(self):
        return self.dataset_itr

    def get_op(self):
        return self.dataset_itr.get_next()
