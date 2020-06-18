import torch as t
try:
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.core.xla_model as xm
except ModuleNotFoundError:
    pass

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
import time


def _mp_fn(index, _hps, _dat_file):
    t.manual_seed(_hps.random_seed)

    # Acquires the (unique) Cloud TPU core corresponding to this process's index
    pre_dev_time = time.time()
    device = xm.xla_device()  
    device_str = xm.xla_real_devices([str(device)])[0]
    elapsed = time.time() - pre_dev_time
    print(f'process {index} acquired {device_str} in {elapsed} seconds',
            file=stderr, flush=True) 

    pre_inst_time = time.time()
    m = ch.Chassis(device, index, _hps, _dat_file)
    print(f'Created Chassis in {time.time() - pre_inst_time:3.5} seconds.', file=stderr, flush=True)
    xm.rendezvous('init')
    m.train()

def run(dat_file, hps='mfcc_inverter,mfcc,train', **kwargs):
    if 'ckpt_file' in kwargs:
        hps = Hyperparams(kwargs)
        if 'random_seed' not in hps:
            hps.random_seed = 2507
    else:
        hps = setup_hparams(hps, kwargs)
        
    netmisc.set_print_iter(0)

    if hps.hw in ('GPU', 'TPU-single'):
        if hps.hw == 'GPU':
            device = t.device('cuda')
            hps.n_loader_workers = 0
        else:
            device = xm.xla_device()
        chs = ch.Chassis(device, 0, hps, dat_file)
        # chs.state.model.print_geometry()
        chs.train()
    elif hps.hw == 'TPU':
        print('Spawning new processes.', file=stderr, flush=True)
        xmp.spawn(_mp_fn, args=(hps, dat_file), nprocs=8, start_method='fork')


if __name__ == '__main__':
    print(sys.executable, ' '.join(arg for arg in sys.argv), file=stderr,
            flush=True)
    fire.Fire(run)

