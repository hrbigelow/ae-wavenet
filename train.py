import sys
from sys import stderr
from pprint import pprint
import torch as t
import fire

import autoencoder_model as ae
import chassis as ch
import parse_tools  
import netmisc
from hparams import setup_hparams, Hyperparams


def _mp_fn(index, _hps, _dat_file):
    m = ch.Chassis(_hps, _dat_file)
    m.train(index)

def run(dat_file, hps='mfcc_inverter,mfcc,train', **kwargs):
    if 'ckpt_file' in kwargs:
        hps = Hyperparams(kwargs)
    else:
        hps = setup_hparams(hps, kwargs)
        
    if hps.hw in ('TPU', 'TPU-single'):
        import torch_xla.distributed.xla_multiprocessing as xmp

    netmisc.set_print_iter(0)

    if hps.hw in ('GPU', 'TPU-single'):
        chs = ch.Chassis(hps, dat_file)
        # chs.state.model.print_geometry()
        chs.train(0)
    elif hps.hw == 'TPU':
        xmp.spawn(_mp_fn, args=(hps, dat_file), nprocs=8)


if __name__ == '__main__':
    fire.Fire(run)

