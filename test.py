# Test trained MFCC inverter model
# Initialize the geometry so that the trim_ups_out is zero


import sys
from sys import stderr
import torch
import parse_tools
import checkpoint
import chassis


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
    elif opts.hwtype == 'CPU':
        pass
    else:
        raise RuntimeError(
                ('Invalid device {} requested.  ' 
                + 'Must be GPU or TPU').format(opts.hwtype))

    print('Using {}'.format(opts.hwtype), file=stderr)
    stderr.flush()

    # generate requested data
    # n_quant = ch.state.model.wavenet.n_quant

    assert opts.hwtype in ('GPU', 'CPU'), 'Currently, Only GPU or CPU supported for sampling'

    if opts.hwtype in ('CPU', 'GPU'):
        chs = chassis.InferenceChassis(mode, opts)
        # chs.state.model.print_geometry()
        chs.infer()
    elif opts.hwtype == 'TPU':
        def _mp_fn(index, mode, opts):
            m = chassis.InferenceChassis(mode, opts)
            m.infer(index)
        xmp.spawn(_mp_fn, args=(mode, opts), nprocs=1, start_method='fork')
    elif opts.hwtype == 'TPU-single':
        chs = chassis.InferenceChassis(mode, opts)
        chs.infer()


if __name__ == '__main__':
    main()

