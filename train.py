import torch as t

import sys
from sys import stderr
from pprint import pprint
import fire
import torch as t

import autoencoder_model as ae
import chassis as ch
import parse_tools  
import netmisc
from hparams import setup_hparams, Hyperparams


def _mp_fn(index, _hps, _dat_file):
    t.manual_seed(_hps.random_seed)

    # Acquires the (unique) Cloud TPU core corresponding to this process's index
    device = xm.xla_device()  
    device_str = xm.xla_real_devices([str(device)])[0]
    print(f'Process {index} is using {device_str}', file=stderr) 

  # Barrier to prevent master from exiting before workers connect.
    m = ch.Chassis(device, index, _hps, _dat_file)
    print(f'Starting training on {device_str}', file=stderr)
    stderr.flush()
    m.train(index)
    xm.rendezvous('init')

def run(dat_file, hps='mfcc_inverter,mfcc,train', **kwargs):
    if 'ckpt_file' in kwargs:
        hps = Hyperparams(kwargs)
    else:
        hps = setup_hparams(hps, kwargs)
        
    if hps.hw in ('TPU', 'TPU-single'):
        import torch_xla.distributed.xla_multiprocessing as xmp

    netmisc.set_print_iter(0)

    if hps.hw in ('GPU', 'TPU-single'):
        if hps.hw == 'GPU':
            device = t.device('cuda')
        else:
            device = xm.xla_device()
        chs = ch.Chassis(device, 0, hps, dat_file)
        # chs.state.model.print_geometry()
        chs.train()
    elif hps.hw == 'TPU':
        xmp.spawn(_mp_fn, args=(hps, dat_file), nprocs=8, start_method='fork')


if __name__ == '__main__':
    fire.Fire(run)

