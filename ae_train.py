import torch
import wave_encoder as we
import mel_mfcc as mm

import ae_model as ae


sample_rate = 16000 # timestep / second
sample_rate_ms = int(sample_rate / 1000) # timestep / ms 
window_length_ms = 25 # ms
hop_length_ms = 10 # ms
n_mels = 80
n_mfcc = 13

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='WaveNet Autoencoder Training')
    parser.add_argument('--resume-step', '-rs', type=int, metavar='INT',
            help='Resume training from '
            + 'CKPT_DIR/<ckpt_pfx>-<resume_step>.')
    parser.add_argument('--cpu-only', '-cpu', action='store_true', default=False,
            help='If present, do all computation on CPU')
    parser.add_argument('--save-interval', '-si', type=int, default=1000, metavar='INT',
            help='Save a checkpoint after this many steps each time')
    parser.add_argument('--progress-interval', '-pi', type=int, default=10, metavar='INT',
            help='Print a progress message at this interval')
    parser.add_argument('--max-steps', '-ms', type=int, default=1e20,
            help='Maximum number of training steps')

    # Training parameter overrides
    parser.add_argument('--batch-size', '-bs', type=int, metavar='INT',
            help='Batch size (overrides PAR_FILE setting)')
    parser.add_argument('--slice-size', '-ss', type=int, metavar='INT',
            help='Slice size (overrides PAR_FILE setting)')
    #parser.add_argument('--l2-factor', '-l2', type=float, metavar='FLOAT',
    #        help='Loss = Xent loss + l2_factor * l2_loss')
    parser.add_argument('--learning-rate', '-lr', type=float, metavar='FLOAT',
            help='Learning rate (overrides PAR_FILE setting)')
    parser.add_argument('--num-global-cond', '-gc', type=int, metavar='INT',
            help='Number of global conditioning categories')

    # positional arguments
    parser.add_argument('ckpt_path', type=str, metavar='CKPT_PATH_PFX',
            help='E.g. /path/to/ckpt/pfx, a path and '
            'prefix combination for writing checkpoint files')
    parser.add_argument('arch_file', type=str, metavar='ARCH_FILE',
            help='JSON file specifying architectural parameters')
    parser.add_argument('par_file', type=str, metavar='PAR_FILE',
            help='JSON file specifying training and other hyperparameters')
    parser.add_argument('sam_file', type=str, metavar='SAMPLES_FILE',
            help='File containing lines:\n'
            + '<id1>\t/path/to/sample1.flac\n'
            + '<id2>\t/path/to/sample2.flac\n')

    return parser.parse_args()



# preprocess
sample_file = '/home/henry/ai/data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac'
mp = mm.MfccProcess()
mda = mp.func(sample_file)

# encoder
kernel_size = 3
in_chan = mda.shape[0]
mid_chan = 768
enc = we.Encoder(in_chan, mid_chan)

print('mda.shape = ', mda.shape)


mda_ten = torch.tensor(mda, dtype=torch.float32)
mda_ten = mda_ten.unsqueeze(0)

latents = enc.forward(mda_ten)

print('latents.shape = ', latents.shape)

# bottleneck
bn_chan = 64
vae = bn.VAE(mid_chan, bn_chan, bias=False)

vae_samples = vae.forward

def main():
    args = get_args()

    import json
    from sys import stderr

    with open(args.arch_file, 'r') as fp:
        arch = json.load(fp)

    with open(args.par_file, 'r') as fp:
        par = json.load(fp)

    # args consistency checks
    if args.num_global_cond is None and 'n_gc_category' not in arch:
        print('Error: must provide n_gc_category in ARCH_FILE, or --num-global-cond',
                file=stderr)
        exit(1)
    
    # Parameter / Arch consistency checks and fixups.
    if args.num_global_cond is not None:
        if args.num_global_cond < dset.get_max_id():
            print('Error: --num-global-cond must be >= {}, the highest ID in the dataset.'.format(
                dset.get_max_id()), file=stderr)
            exit(1)
        else:
            arch['n_gc_category'] = args.num_global_cond
    
    # Construct model
    model = ae.AutoEncoder()

    # Set CPU or GPU context

    # Restore from checkpoint
    if args.resume_step:
        pass
        print('Restored net and dset from checkpoint', file=stderr)

    # Initialize optimizer

    # Start training
    print('Starting training...', file=stderr)
    step = args.resume_step or 1
    while step < args.max_steps:

    if step % args.save_interval == 0 and step != args.resume_step:
        net_save_path = net.save(step)
        dset_save_path = dset.save(step, file_read_count)
        print('Saved checkpoints to {} and {}'.format(net_save_path, dset_save_path),
                file=stderr)

    step += 1

if __name__ == '__main__':
    main()


