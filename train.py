import sys
from sys import stderr
from pprint import pprint
import torch as t
import fire

import autoencoder_model as ae
import chassis as ch
import parse_tools  
import netmisc
from hparams import setup_hparams


def run(dat_file, hps='mfcc_inverter,mfcc,train', **kwargs):
    hps = setup_hparams(hps, kwargs)
    if hps.hw == 'GPU':
        if not t.cuda.is_available():
            raise RuntimeError('GPU requested but not available')
    elif hps.hw in ('TPU', 'TPU-single'):
        import torch_xla.distributed.xla_multiprocessing as xmp
    else:
        raise RuntimeError(
                ('Invalid device {} requested.  ' 
                + 'Must be GPU or TPU').format(hps.hw))

    print('Hyperparameters:\n', '\n'.join(f'{k} = {v}' for k, v in hps.items()), file=stderr)
    print(f'Using {hps.hw}', file=stderr)

    netmisc.set_print_iter(0)

    if hps.hw in ('GPU', 'TPU-single'):
        chs = ch.Chassis(hps, dat_file)
        # chs.state.model.print_geometry()
        chs.train(hps, 0)
    elif hps.hw == 'TPU':
        def _mp_fn(index):
            m = ch.Chassis(hps, dat_file)
            m.train(hps, index)
        xmp.spawn(_mp_fn, args=(), nprocs=8, start_method='fork')


if __name__ == '__main__':
    fire.Fire(run)

