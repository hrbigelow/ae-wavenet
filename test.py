# Test trained MFCC inverter model
# Initialize the geometry so that the trim_ups_out is zero


import sys
from sys import stderr
import torch as t
import fire
import parse_tools
import checkpoint
import chassis
from hparams import setup_hparams, Hyperparams


def run(dat_file, hps='mfcc_inverter,mfcc,test', **kwargs):
    hps = setup_hparams(hps, kwargs)
    assert hps.hw in ('GPU', 'CPU'), 'Currently, Only GPU or CPU supported for sampling'

    if 'random_seed' not in hps:
        hps.random_seed = 2507

    if hps.hw == 'GPU':
        if not t.cuda.is_available():
            raise RuntimeError('GPU requested but not available')
    # elif hps.hw in ('TPU', 'TPU-single'):
        # import torch_xla.distributed.xla_multiprocessing as xmp
    elif hps.hw == 'CPU':
        pass
    else:
        raise RuntimeError(
                ('Invalid device {} requested.  ' 
                + 'Must be GPU or TPU').format(hps.hw))

    print('Using {}'.format(hps.hw), file=stderr)
    stderr.flush()

    # generate requested data
    # n_quant = ch.state.model.wavenet.n_quant


    if hps.hw in ('CPU', 'GPU'):
        if hps.hw == 'GPU':
            device = t.device('cuda')
            hps.n_loader_workers = 0
        else:
            device = t.device('cpu')

        chs = chassis.InferenceChassis(device, 0, hps, dat_file)
        if hps.jit_script_path:
            # data_scr = t.jit.script(chs.state.data_loader.dataset)
            model_scr = t.jit.script(chs.state.model.wavenet)
            model_scr.save(hps.jit_script_path)
            model_scr.to(chs.device)
            # print(model_scr.code)
            print('saved {}'.format(hps.jit_script_path))
            chs.infer(model_scr)
            return

        # chs.state.model.print_geometry()
        chs.infer()
    # elif hps.hw == 'TPU':
        # def _mp_fn(index, mode, hps):
            # m = chassis.InferenceChassis(mode, hps)
            # m.infer(index)
        # xmp.spawn(_mp_fn, args=(mode, hps), nprocs=1, start_method='fork')
    # elif hps.hw == 'TPU-single':
        # chs = chassis.InferenceChassis(mode, hps)
        # chs.infer()


if __name__ == '__main__':
    print(sys.executable, ' '.join(arg for arg in sys.argv), file=stderr,
            flush=True)
    fire.Fire(run)

