from sys import stderr
import argparse
import data

def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--n-quant', '-nq', type=int, metavar='INT',
            default=256, help='Number of quantization levels for Mu-law companding')
    p.add_argument('--sample-rate', '-sr', type=int, metavar='INT',
            default=16000, help='Number of samples per second for parsing sound files')
    # positional arguments
    p.add_argument('sam_file', type=str, metavar='SAMPLES_FILE',
            help='File containing lines:\n'
            + '<id1>\t/path/to/sample1.flac\n'
            + '<id2>\t/path/to/sample2.flac\n')
    p.add_argument('dat_file', type=str, metavar='DAT_FILE',
            help='Specify prefix for creating {dat_file}')
    return p

def main():
    parser = make_parser()
    opts = parser.parse_args()

    print('Starting...', file=stderr)
    stderr.flush()

    catalog = data.parse_catalog(opts.sam_file)
    data.convert(catalog, opts.dat_file, opts.n_quant, opts.sample_rate)
    print('Wrote catalog to {}'.format(opts.dat_file),
            file=stderr)
    return 0


if __name__ == '__main__':
    main()

    
