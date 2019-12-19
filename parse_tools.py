import argparse

top_usage = """
Usage: train.py {new|resume} [options]

train.py new [options] 
   -- train a new model
train.py resume [options]
   -- resume training from .ckpt file 
"""

# Training options common to both "new" and "resume" training modes
def train_parser():
    train = argparse.ArgumentParser(add_help=False)
    train.add_argument('--n-batch', '-nb', type=int, metavar='INT',
            default=16, help='Batch size')
    train.add_argument('--n-win-batch', '-nw', type=int, metavar='INT',
            default=100, help='# of consecutive window samples in one slice' )
    train.add_argument('--max-steps', '-ms', type=int, metavar='INT', default=1e20,
            help='Maximum number of training steps')
    train.add_argument('--save-interval', '-si', type=int, default=1000, metavar='INT',
            help='Save a checkpoint after this many steps each time')
    train.add_argument('--progress-interval', '-pi', type=int, default=1, metavar='INT',
            help='Print a progress message at this interval')
    train.add_argument('--hwtype', '-hw', type=str, default='GPU',
            help='Harware target, one of CPU, GPU, or TPU')
    train.add_argument('--learning-rate-steps', '-lrs', type=int, nargs='+',
            metavar='INT', default=[0, 4e6, 6e6, 8e6],
            help='Learning rate starting steps to apply --learning-rate-rates')
    train.add_argument('--learning-rate-rates', '-lrr', type=float, nargs='+',
            metavar='FLOAT', default=[4e-4, 2e-4, 1e-4, 5e-5],
            help='Each of these learning rates will be applied at the '
            'corresponding value for --learning-rate-steps')
    train.add_argument('--random-seed', '-rnd', type=int, metavar='INT',
            default=2507,
            help='Random seed for weights initialization etc')
    train.add_argument('ckpt_template', type=str, metavar='CHECKPOINT_TEMPLATE',
            help="Full or relative path, including a filename template, containing "
            "a single %%, which will be replaced by the step number.")
    # VAE-specific Bottleneck
    train.add_argument('--bn-free-nats', '-fn', type=int, metavar='INT',
            default=9, help='number of free nats in KL divergence that are '
            'not penalized')
    train.add_argument('--bn-anneal-weight-steps', '-aws', type=int, nargs='+',
            metavar='INT', default=[0, 2e3, 4e3, 6e3, 8e3, 1e4, 2e4, 3e4, 4e4,
                5e4, 6e4],
            help='Learning rate starting steps to apply --anneal-weight-vals')
    train.add_argument('--bn-anneal-weight-vals', '-awv', type=float, nargs='+',
            metavar='FLOAT', default=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                0.8, 0.9, 1.0],
            help='Each of these anneal weights will be applied at the '
            'corresponding step for --anneal-weight-steps')

    return train

# Complete parser for cold-start mode
def cold_parser():
    tp = train_parser()
    cold = argparse.ArgumentParser(parents=[tp])

    cold.add_argument('--arch-file', '-af', type=str, metavar='ARCH_FILE',
            help='INI file specifying architectural parameters')
    cold.add_argument('--train-file', '-tf', type=str, metavar='TRAIN_FILE',
            help='INI file specifying training and other hyperparameters')

    # Preprocessing parameters
    cold.add_argument('--pre-sample-rate', '-sr', type=int, metavar='INT', default=16000,
            help='# samples per second in input wav files')
    cold.add_argument('--pre-mfcc-win-sz', '-wl', type=int, metavar='INT', default=400,
            help='size of the MFCC window length in timesteps')
    cold.add_argument('--pre-mfcc-hop-sz', '-hl', type=int, metavar='INT', default=160,
            help='size of the hop length for MFCC preprocessing, in timesteps')
    cold.add_argument('--pre-n-mels', '-nm', type=int, metavar='INT', default=80,
            help='number of mel frequency values to calculate')
    cold.add_argument('--pre-n-mfcc', '-nf', type=int, metavar='INT', default=13,
            help='number of mfcc values to calculate')
    cold.prog += ' new'

    # Encoder architectural parameters
    cold.add_argument('--enc-n-out', '-no', type=int, metavar='INT', default=768,
            help='number of output channels')

    # Bottleneck architectural parameters
    cold.add_argument('--bn-type', '-bt', type=str, metavar='STR', default='ae',
            help='bottleneck type (one of "ae", "vae", or "vqvae")')
    cold.add_argument('--bn-n-out', '-bo', type=int, metavar='INT', default=64,
            help='number of output channels for the bottleneck')
    cold.add_argument('--bn-vq-gamma', '-vqb', type=float, metavar='FLOAT', default=0.25,
            help='beta multiplier for commitment loss term, Eq 3 from Chorowski et al.')
    cold.add_argument('--bn-vq-n-embed', '-vqn', type=int, metavar='INT', default=4096,
            help='number of embedding vectors, K, in section 3.1 of VQVAE paper')


    # Decoder architectural parameters
    cold.add_argument('--dec-jitter-prob', '-djp', type=float, metavar='FLOAT',
            default=0.12,
            help='replacement probability for time-jitter regularization')
    cold.add_argument('--dec-filter-sz', '-dfs', type=int, metavar='INT', default=2,
            help='decoder number of dilation kernel elements')
    # !!! This is set equal to --bn-n-out
    #cold.add_argument('--dec-n-lc-in', '-dli', type=int, metavar='INT', default=-1,
    #        help='decoder number of local conditioning input channels')
    cold.add_argument('--dec-n-lc-out', '-dlo', type=int, metavar='INT', default=-1,
            help='decoder number of local conditioning output channels')
    cold.add_argument('--dec-n-res', '-dnr', type=int, metavar='INT', default=-1,
            help='decoder number of residual channels')
    cold.add_argument('--dec-n-dil', '-dnd', type=int, metavar='INT', default=-1,
            help='decoder number of dilation channels')
    cold.add_argument('--dec-n-skp', '-dns', type=int, metavar='INT', default=-1,
            help='decoder number of skip channels')
    cold.add_argument('--dec-n-post', '-dnp', type=int, metavar='INT', default=-1,
            help='decoder number of post-processing channels')
    cold.add_argument('--dec-n-quant', '-dnq', type=int, metavar='INT', 
            help='decoder number of input channels')
    cold.add_argument('--dec-n-blocks', '-dnb', type=int, metavar='INT',
            help='decoder number of dilation blocks')
    cold.add_argument('--dec-n-block-layers', '-dnl', type=int, metavar='INT', 
            help='decoder number of power-of-two dilated '
            'convolutions in each layer')
    cold.add_argument('--dec-n-global-embed', '-dng', type=int, metavar='INT',
            help='decoder number of global embedding channels')

    # MFCC parameters
    cold.add_argument('--win-size', '-ws', type=int, metavar='INT',
            default=400,
            help='Number of timesteps used to calculate MFCC coefficients')
    cold.add_argument('--hop-size', '-hs', type=int, metavar='INT',
            default=160,
            help='Number of timesteps to hop between consecutive MFCC coefficients')

    # positional arguments
    cold.add_argument('dat_file', type=str, metavar='DAT_FILE',
            help='File created by preprocess.py')
    return cold

# Complete parser for resuming from Checkpoint
def resume_parser():
    tp = train_parser()
    resume = argparse.ArgumentParser(parents=[tp], add_help=True)
    resume.add_argument('ckpt_file', type=str, metavar='CHECKPOINT_FILE',
            help="""Checkpoint file generated from a previous run.  Restores model
            architecture, model parameters, and data generator state.""")
    resume.add_argument('dat_file', type=str, metavar='DAT_FILE',
            help='File created by preprocess.py')
    resume.prog += ' resume'
    return resume

def two_stage_parse(cold_parser, args=None):  
    '''wrapper for parse_args for overriding options from file'''
    default_opts = cold_parser.parse_args(args)

    cli_parser = argparse.ArgumentParser(parents=[cold_parser], add_help=False)
    dests = {co.dest:argparse.SUPPRESS for co in cli_parser._actions}
    cli_parser.set_defaults(**dests)
    cli_parser._defaults = {} # hack to overcome bug in set_defaults
    cli_opts = cli_parser.parse_args(args)

    # Each option follows the rule:
    # Use JSON file setting if present.  Otherwise, use command-line argument,
    # Otherwise, use command-line default
    import json
    try:
        with open(cli_opts.arch_file) as fp:
            arch_opts = json.load(fp)
    except AttributeError:
        arch_opts = {} 
    except FileNotFoundError:
        print("Error: Couldn't open arch parameters file {}".format(cli_opts.arch_file))
        exit(1)

    try:
        with open(cli_opts.train_file) as fp:
            train_opts = json.load(fp)
    except AttributeError:
        train_opts = {}
    except FileNotFoundError:
        print("Error: Couldn't open train parameters file {}".format(cli_opts.train_file))
        exit(1)

    # Override with command-line settings, then defaults
    merged_opts = vars(default_opts)
    merged_opts.update(arch_opts)
    merged_opts.update(train_opts)
    merged_opts.update(vars(cli_opts))

    # Convert back to a Namespace object
    return argparse.Namespace(**merged_opts) 
    # return cli_opts


def get_prefixed_items(d, pfx):
    '''select all items whose keys start with pfx, and strip that prefix'''
    return { k[len(pfx):]:v for k,v in d.items() if k.startswith(pfx) }

