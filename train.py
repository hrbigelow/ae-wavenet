import torch
import wave_encoder as we
import model as ae
import data as D 


def get_opts():
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--arch-file', '-af', type=str, metavar='ARCH_FILE',
            help='INI file specifying architectural parameters')
    parser.add_argument('--train-file', '-pf', type=str, metavar='PAR_FILE',
            help='INI file specifying training and other hyperparameters')

    # Parameters in arch file will be overridden by any given here.

    # Preprocessing parameters
    parser.add_argument('--pre-sample-rate', '-sr', type=int, default=16000,
            help='# samples per second in input wav files')
    parser.add_argument('--pre-win-sz', '-wl', type=int, default=400,
            help='size of the MFCC window length in timesteps')
    parser.add_argument('--pre-hop-sz', '-hl', type=int, default=160,
            help='size of the hop length for MFCC preprocessing, in timesteps')
    parser.add_argument('--pre-n-mels', '-nm', type=int, default=80,
            help='number of mel frequency values to calculate')
    parser.add_argument('--pre-n-mfcc', '-nf', type=int, default=13,
            help='number of mfcc values to calculate')

    # Encoder architectural parameters
    parser.add_argument('--enc-n-out', '-no', type=int, default=768,
            help='number of output channels')

    # Bottleneck architectural parameters
    parser.add_argument('--bn-type', '-bt', type=str, default='ae',
            help='bottleneck type (one of "ae", "vae", or "vqvae")')
    parser.add_argument('--bn-n-out', '-bo', type=int, default=64,
            help='number of output channels for the bottleneck')

    # Decoder architectural parameters
    parser.add_argument('--dec-filter-sz', '-dfs', type=int, default=2,
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
    parser.add_argument('--dec-n-blocks', '-dnb', type=int,
            help='decoder number of dilation blocks')
    parser.add_argument('--dec-n-block-layers', '-dnl', type=int, 
            help='decoder number of power-of-two dilated '
            'convolutions in each layer')
    parser.add_argument('--dec-n-global-embed', '-dng', type=int,
            help='decoder number of global embedding channels')

    # Training parameters
    parser.add_argument('--n-batch', '-nb', type=int, metavar='INT',
            help='Batch size')
    parser.add_argument('--n-sam-per-slice', '-nw', type=int, metavar='INT',
            default=100,
            help='# of consecutive window samples in one slice' )
    parser.add_argument('--frac-permutation-use', '-fpu', type=float,
            metavar='FLOAT', help='Fraction of each random data permutation to '
            'use.  Lower fraction causes more frequent reading of data from '
            'disk, but more globally random order of data samples presented '
            'to the model')
    parser.add_argument('--requested-wav-buf-sz', '-rws', type=int,
            metavar='INT', help='Size in bytes of the total memory available '
            'to buffer training data.  A higher value will minimize re-reading '
            'of data and allow more globally random sample order')
    parser.add_argument('--max-steps', '-ms', type=int, default=1e20,
            help='Maximum number of training steps')
    parser.add_argument('--resume-step', '-rst', type=int, default=0,
            help='Step to resume training.  If > 0, the checkpoint file is '
            'determined from ckpt_path and --resume-step')
    parser.add_argument('--save-interval', '-si', type=int, default=1000, metavar='INT',
            help='Save a checkpoint after this many steps each time')
    parser.add_argument('--progress-interval', '-pi', type=int, default=10, metavar='INT',
            help='Print a progress message at this interval')
    parser.add_argument('--disable-cuda', '-dc', action='store_true', default=False,
            help='If present, do all computation on CPU')
    parser.add_argument('--learning-rate-steps', '-lrs', type=int, nargs='+',
            metavar='INT', help='Learning rate starting steps to apply --learning-rate-rates')
    parser.add_argument('--learning-rate-rates', '-lrr', type=float, nargs='+',
            metavar='FLOAT', help='Each of these learning rates will be applied at the '
            'corresponding value for --learning-rate-steps')

    # positional arguments
    parser.add_argument('sam_file', type=str, metavar='SAMPLES_FILE',
            help='File containing lines:\n'
            + '<id1>\t/path/to/sample1.flac\n'
            + '<id2>\t/path/to/sample2.flac\n')
    parser.add_argument('checkpoint_dir', type=str, metavar='CHECKPOINT_DIR',
            help='Directory for writing checkpoint files')
    parser.add_argument('checkpoint_template', type=str,
            metavar='CHECKPOINT_TEMPLATE',
            help='Filename template, with one "{}" for step number, '
            'for writing checkpoint files')

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

    opts.device = None
    if not opts.disable_cuda and torch.cuda.is_available():
        opts.device = torch.device('cuda')
    else:
        opts.device = torch.device('cpu') 

    from sys import stderr
    
    # Construct model
    preprocess_params = get_prefixed_items(vars(opts), 'pre_')
    encoder_params = get_prefixed_items(vars(opts), 'enc_')
    bn_params = get_prefixed_items(vars(opts), 'bn_')
    decoder_params = get_prefixed_items(vars(opts), 'dec_')

    # Prepare data
    data = D.WavSlices(opts.sam_file, preprocess_params['sample_rate'],
            opts.frac_permutation_use)

    data_ckpt_file_template = opts.checkpoint_template + '.data.ckpt'
    model_ckpt_file_template = opts.checkpoint_template + '.model.ckpt'

    data.ckpt_path.enable(opts.checkpoint_dir, data_ckpt_file_template)

    decoder_params['n_speakers'] = data.num_speakers()
    model = ae.AutoEncoder(preprocess_params, encoder_params, bn_params,
            decoder_params, opts.device)
    model.ckpt_path.enable(opts.checkpoint_dir, model_ckpt_file_template)

    # the receptive_field is the length of one logical sample.  the data module
    # yields n_batch * n_sam_per_slice logical samples at a time.  since the
    # logical samples from one .wav file are overlapping, this amounts to a
    # window of n_win + receptive_field_sz - 1 from each of the n_batch wav
    # files.
    model.set_geometry(opts.n_sam_per_slice)

    data.set_geometry(opts.n_batch, model.input_size, model.output_size,
            opts.requested_wav_buf_sz)

    model.to(device=model.device)

    #total_bytes = 0
    #for name, par in model.named_parameters():
    #    n_bytes = par.data.nelement() * par.data.element_size()
    #    total_bytes += n_bytes
    #    print(name, type(par.data), par.size(), n_bytes)
    #print('total_bytes: ', total_bytes)

    # Restore from checkpoint, or initialize model parameters
    step = opts.resume_step or 0
    if step > 0:
        data.ckpt_to_state(data.file_to_ckpt(step))
        model.load_state_dict(torch.load(model.ckpt_path.path(step)))
        print('Restored model and data from checkpoint', file=stderr)
    else:
        print('Initializing model parameters', file=stderr)
        model.initialize_weights()

    # Initialize optimizer
    model_params = model.parameters()
    loss_fcn = model.loss_factory(data.batch_slice_gen_fn())

    # Start training
    print('Starting training...', file=stderr)

    learning_rates = dict(zip(opts.learning_rate_steps, opts.learning_rate_rates))
    while step < opts.max_steps:
        if step in learning_rates:
            optim = torch.optim.Adam(params=model_params, lr=learning_rates[step])
        loss = optim.step(loss_fcn)

        if step % opts.progress_interval == 0:
            print('Step {}, Loss: {}'.format(step, loss))

        if step % opts.save_interval == 0 and step != opts.resume_step:
            torch.save(model.state_dict(), model.ckpt_path.path(step))
            data.ckpt_to_file(data.state_to_ckpt(), step)
            print('Saved checkpoints to {} and {}'.format(data.ckpt_path.path(step),
                model.ckpt_path.path(step)),
                file=stderr)

        step += 1

if __name__ == '__main__':
    main()

