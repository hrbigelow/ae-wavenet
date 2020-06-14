# All keys that appear in each entry of HPARAMS_REGISTRY must also appear in
# some entry of DEFAULTS
HPARAMS_REGISTRY = {}
DEFAULTS = {}

class Hyperparams(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)


def setup_hparams(hparam_set_names, kwargs):
    H = Hyperparams()
    if not isinstance(hparam_set_names, tuple):
        hparam_set_names = hparam_set_names.split(",")
    hparam_sets = [HPARAMS_REGISTRY[x.strip()] for x in hparam_set_names if x] + [kwargs]
    for k, v in DEFAULTS.items():
        H.update(v)
    for hps in hparam_sets:
        for k in hps:
            if k not in H:
                raise ValueError(f"{k} not in default args")
        H.update(**hps)
    H.update(**kwargs)
    return H


mfcc = Hyperparams(
    sample_rate = 16000,
    mfcc_win_sz = 400,
    mfcc_hop_sz = 160,
    n_mels = 80,
    n_mfcc = 13,
    n_lc_in = 39
)
        
HPARAMS_REGISTRY["mfcc"] = mfcc
DEFAULTS["mfcc"] = mfcc

wavenet = Hyperparams(
    filter_sz = 2,
    n_lc_out = 128,
    lc_upsample_strides = [5, 4, 4, 2],
    lc_upsample_filt_sizes = [25, 16, 16, 16],
    n_res = 368,
    n_dil = 256,
    n_skp = 256,
    n_post = 256,
    n_quant = 256,
    n_blocks = 2,
    n_block_layers = 10,
    n_global_embed = 10,
    n_speakers = 40, 
    jitter_prob = 0.0,
    free_nats = 9,
    bias = True
)


HPARAMS_REGISTRY["wavenet"] = wavenet
DEFAULTS["wavenet"] = wavenet

mfcc_inverter = Hyperparams(
    global_model = 'mfcc_inverter'
)

mfcc_inverter.update(wavenet)
HPARAMS_REGISTRY['mfcc_inverter'] = mfcc_inverter
DEFAULTS['mfcc_inverter'] = mfcc_inverter

train_tpu = Hyperparams(
    hw = 'TPU',
    n_batch = 16,
    n_win_batch = 5000,
    n_epochs = 10,
    save_interval = 1000,
    progress_interval = 1,
    skip_loop_body = False,
    n_loader_workers = 4,
    log_dir = '/tmp',
    random_seed = 2507,
    learning_rate_steps = [ 0, 4e6, 6e6, 8e6 ],
    learning_rate_rates = [ 1e-4, 5e-5, 5e-5, 5e-5 ],
    ckpt_template = '%.ckpt',
    ckpt_file = None
)

HPARAMS_REGISTRY["train"] = train_tpu
DEFAULTS["train"] = train_tpu


