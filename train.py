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

import psutil

def _clear_stray_processes():
  curr_proc = psutil.Process()
  curr_cmdline = list(curr_proc.cmdline())
  for proc in psutil.process_iter():
    if proc.pid != curr_proc.pid and proc.cmdline() == curr_cmdline:
      proc.kill()


def _mp_fn(index, _hps, _dat_file):
    t.manual_seed(_hps.random_seed)

    # Acquires the (unique) Cloud TPU core corresponding to this process's index
    pre_dev_time = time.time()
    print(f'Replica {index} acquiring a device...', end='', file=stderr)
    device = xm.xla_device()  
    device_str = xm.xla_real_devices([str(device)])[0]
    elapsed = time.time() - pre_dev_time
    print(f'process {index} acquired {device_str} in {elapsed} seconds', file=stderr) 
    stderr.flush()

    pre_inst_time = time.time()
    print(f'Replica {index} instantiating Chassis...', end='', file=stderr)
    m = ch.Chassis(device, index, _hps, _dat_file)
    elapsed = time.time() - pre_inst_time
    print(f'done in {elapsed} seconds.', file=stderr)
    stderr.flush()
    xm.rendezvous('init')
    m.train()

def run(dat_file, hps='mfcc_inverter,mfcc,train', **kwargs):
    if 'ckpt_file' in kwargs:
        hps = Hyperparams(kwargs)
        if 'random_seed' not in hps:
            hps.random_seed = 2507
    else:
        hps = setup_hparams(hps, kwargs)
        
    if hps.hw in ('TPU', 'TPU-single'):
        import torch_xla.distributed.xla_multiprocessing as xmp
        import torch_xla.core.xla_model as xm

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
        print('Spawning new processes.', file=stderr)
        stderr.flush()
        _clear_stray_processes()
        xmp.spawn(_mp_fn, args=(hps, dat_file), nprocs=8, start_method='fork')


if __name__ == '__main__':
    fire.Fire(run)

