import data
import pickle

sample_rate = 16000
frac_perm_use = 0.1
req_wav_buf_sz = 1e7
sam_file = '/home/henry/ai/data/librispeech.dev-clean.rdb'
n_batch = 4
input_size = 4000
output_size = 2000

sample_catalog = data.parse_sample_catalog(sam_file)
dwav = data.WavSlices(sample_catalog, sample_rate, frac_perm_use, req_wav_buf_sz)
dwav.set_geometry(n_batch, input_size, output_size)

batch_gen = dwav.batch_slice_gen_fn()

for i in range(1000):
    __, voice_inds, wav = next(batch_gen)
    # print(dwav, end='')

dwav_state = pickle.dumps(dwav)
print('Yielding: ', dwav, end='')
__, voice_inds, wav = next(batch_gen)

dwav_r = pickle.loads(dwav_state)
print('Restored: ', dwav_r, end='')
dwav_r.set_geometry(n_batch, input_size, output_size)

batch_gen_r = dwav_r.batch_slice_gen_fn()
print('Yielding: ', dwav_r, end='')
__, voice_inds_r, wav_r = next(batch_gen_r)

assert (voice_inds_r == voice_inds).all()
assert (wav_r == wav).all()

