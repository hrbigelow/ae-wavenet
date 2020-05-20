# Test trained MFCC inverter model
# Initialize the geometry so that the trim_ups_out is zero

"""
Start with 20 Mel vectors, which is the number required to have at least
one output wav timestep.  However,
"""

# Initialize model with a "white noise seed", consisting of white noise wav
# content, together with the MFCC's computed for that white noise.

# Initialize
# Rotate the MFCC input tensor (add one to the end, take one off the beginning)
# Rotate WAV tensor (subtract 160 from beginning)
# Set trim_ups_out to 159

# Loop 160 iterations
# Run input
# Sample from output
# Rotate input wav tensor by 1, adding sampled output to end
# Advance trim_ups_out by 1




import sys
from sys import stderr
import torch
import parse_tools


def main():
    if len(sys.argv) == 1 or sys.argv[1] not in ('inverter'):
        print(parse_tools.test_usage, file=stderr)
        return

    mode = sys.argv[1]
    del sys.argv[1]

    if mode == 'inverter':
        inv_parser = parse_tools.wav_gen_parser()
        opts = parse_tools.two_stage_parse(inv_parser)

    if opts.hwtype == 'GPU':
        if not torch.cuda.is_available():
            raise RuntimeError('GPU requested but not available')
    elif opts.hwtype in ('TPU', 'TPU-single'):
        import torch_xla.distributed.xla_multiprocessing as xmp
    else:
        raise RuntimeError(
                ('Invalid device {} requested.  ' 
                + 'Must be GPU or TPU').format(opts.hwtype))

    print('Using {}'.format(opts.hwtype), file=stderr)
    stderr.flush()

    # load the trained model
    state = checkpoint.InferenceState()
    state.load(opts.ckpt_file, opts.dat_file)

    # generate requested data
    n_quant = state.model.n_quant

    assert opts.hwtype == 'GPU', 'Currently, Only GPU supported for sampling'

    for mb in state.data_loader:
        init_wav = mb.wav_enc_input[:wav_in_end]
        wav_sample = state.model.sample(init_wav, mb.mel_enc_input,
                mb.voice_index, mb.jitter_index, opts.n_sample_replicas)

        # save results to specified files
        for i in opts.n_rep:
            wav_final = util.mu_decode_torch(wav_sample[i], n_quant)
            with open(opts.out_template.format(mb.voice_index, i)) as fh:
                torch.save(wav_final, fh)

    if opts.hwtype == 'GPU':
        pass
    elif opts.hwtype == 'TPU-single':
        pass

if __name__ == '__main__':
    main()

