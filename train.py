import torch
import wave_encoder as we
import model as ae
import data as D 


def get_opts():
    import argparse

    parser = argparse.ArgumentParser(add_help=False, )
    parser.add_argument('--arch-file', '-af', type=str, metavar='ARCH_FILE',
            help='INI file specifying architectural parameters')
    parser.add_argument('--train-file', '-pf', type=str, metavar='PAR_FILE',
            help='INI file specifying training and other hyperparameters')

    # Parameters in arch file will be overridden by any given here.

    # Encoder architectural parameters
    parser.add_argument('--enc-sample-rate-ms', '-sr', type=int, default=16,
            help='# samples per millisecond in input wav files')
    parser.add_argument('--enc-win-length-ms', '-wl', type=int, default=25,
            help='size of the MFCC window length in milliseconds')
    parser.add_argument('--enc-hop-length-ms', '-hl', type=int, default=10,
            help='size of the hop length for MFCC preprocessing, in milliseconds')
    parser.add_argument('--enc-n-mels', '-nm', type=int, default=80,
            help='number of mel frequency values to calculate')
    parser.add_argument('--enc-n-mfcc', '-nf', type=int, default=13,
            help='number of mfcc values to calculate')
    parser.add_argument('--enc-n-out', '-no', type=int, default=768,
            help='number of output channels')

    # Bottleneck architectural parameters
    parser.add_argument('--bn-type', '-bt', type=str, default='ae',
            help='bottleneck type (one of "ae", "vae", or "vqvae")')
    parser.add_argument('--bn-n-out', '-bo', type=int, default=64,
            help='number of output channels for the bottleneck')

    # Decoder architectural parameters
    parser.add_argument('--dec-n-kern', '-dnk', type=int, default=2,
            help='decoder number of dilation kernel elements')
    parser.add_argument('--dec-n-lc-in', '-dli', type=int, default=-1,
            help='decoder number of local conditioning input channels')
    parser.add_argument('--dec-n-lc-out', '-dlo', type=int, default=-1,
            help='decoder number of local conditioning output channels')
    parser.add_argument('--dec-n-res', '-dnr', type=int, default=-1,
            help='decoder number of residual channels')
    parser.add_argument('--dec-n-dil', '-dnd', type=int, default=-1,
            help='decoder number of dilation channels')
    parser.add_argument('--dec-n-skp', '-dns', type=int, default=-1,
            help='decoder number of skip channels')
    parser.add_argument('--dec-n-post', '-dnp', type=int, default=-1,
            help='decoder number of post-processing channels')
    parser.add_argument('--dec-n-quant', '-dnq', type=int,
            help='decoder number of input channels')

    # Training parameters
    parser.add_argument('--n-batch', '-nb', type=int, metavar='INT',
            help='Batch size')
    parser.add_argument('--n-win', '-nw', type=int, metavar='INT',
            help='# of consecutive window training samples in one batch channel' )
    parser.add_argument('--learning-rate', '-lr', type=float, metavar='FLOAT',
            help='Learning rate')
    #parser.add_argument('--l2-factor', '-l2', type=float, metavar='FLOAT',
    #        help='Loss = Xent loss + l2_factor * l2_loss')
    parser.add_argument('--cpu-only', '-cpu', action='store_true', default=False,
            help='If present, do all computation on CPU')
    parser.add_argument('--save-interval', '-si', type=int, default=1000, metavar='INT',
            help='Save a checkpoint after this many steps each time')
    parser.add_argument('--progress-interval', '-pi', type=int, default=10, metavar='INT',
            help='Print a progress message at this interval')
    parser.add_argument('--max-steps', '-ms', type=int, default=1e20,
            help='Maximum number of training steps')

    # positional arguments
    parser.add_argument('ckpt_path', type=str, metavar='CKPT_PATH_PFX',
            help='E.g. /path/to/ckpt/pfx, a path and '
            'prefix combination for writing checkpoint files')
    parser.add_argument('sam_file', type=str, metavar='SAMPLES_FILE',
            help='File containing lines:\n'
            + '<id1>\t/path/to/sample1.flac\n'
            + '<id2>\t/path/to/sample2.flac\n')

    default_opts = parser.parse_args()

    cli_parser = argparse.ArgumentParser(parents=[parser])
    dests = {co.dest:argparse.SUPPRESS for co in cli_parser._actions}
    cli_parser.set_defaults(**dests)
    cli_parser._defaults = {} # hack to overcome bug in set_defaults
    cli_opts = cli_parser.parse_args()

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


def main():
    opts = get_opts()

    from sys import stderr
    
    # Construct model
    encoder_params = get_prefixed_items(vars(opts), 'enc_')
    bn_params = get_prefixed_items(vars(opts), 'bn_')
    decoder_params = get_prefixed_items(vars(opts), 'dec_')

    # Prepare data
    data = D.WavSlices(opts.sam_file, opts.n_win, opts.n_batch,
            encoder_params['sample_rate_ms'] * 1000, opts.frac_permutation_use,
            opts.requested_wav_buf_sz)

    decoder_params['n_speakers'] = data.n_speakers()

    model = ae.AutoEncoder(encoder_params, bn_params, decoder_params)

    data.set_receptive_field(model.get_receptive_field())

    # Set CPU or GPU context

    # Restore from checkpoint
    if opts.resume_step:
        pass
        print('Restored net and dset from checkpoint', file=stderr)

    # Initialize optimizer

    # Start training
    print('Starting training...', file=stderr)
    step = opts.resume_step or 1
    while step < opts.max_steps:

        if step % opts.save_interval == 0 and step != opts.resume_step:
            net_save_path = net.save(step)
            dset_save_path = dset.save(step, file_read_count)
            print('Saved checkpoints to {} and {}'.format(net_save_path, dset_save_path),
                    file=stderr)

        step += 1

if __name__ == '__main__':
    main()




