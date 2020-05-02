import sys
from sys import stderr
from pprint import pprint
import torch

import autoencoder_model as ae
import chassis as ch
import parse_tools  
import netmisc


def main():
    if len(sys.argv) == 1 or sys.argv[1] not in ('new', 'resume'):
        print(parse_tools.top_usage, file=stderr)
        return

    print('Command line: ', ' '.join(sys.argv), file=stderr)
    stderr.flush()

    mode = sys.argv[1]
    del sys.argv[1]
    if mode == 'new':
        cold_parser = parse_tools.cold_parser()
        opts = parse_tools.two_stage_parse(cold_parser)
    elif mode == 'resume':
        resume_parser = parse_tools.resume_parser()
        opts = resume_parser.parse_args()  

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

    # Start training
    print('Training parameters used:', file=stderr)
    pprint(opts, stderr)

    # set this to zero if you want to print out a logging header in resume mode as well
    netmisc.set_print_iter(0)

    if opts.hwtype == 'GPU':
        chs = ch.Chassis(mode, opts)
        # chs.state.model.print_geometry()
        chs.train(0)
    elif opts.hwtype == 'TPU':
        def _mp_fn(index, mode, opts):
            m = ch.Chassis(mode, opts)
            m.train(index)
        xmp.spawn(_mp_fn, args=(mode, opts), nprocs=1, start_method='fork')
    elif opts.hwtype == 'TPU-single':
        ch.Chassis(mode, opts).train(0)


if __name__ == '__main__':
    main()

