import sys
from sys import stderr
from pprint import pprint
import torch

import model as ae
import parse_tools  
import netmisc


def main():
    if len(sys.argv) == 1 or sys.argv[1] not in ('new', 'resume'):
        print(parse_tools.top_usage, file=stderr)
        return

    print('Command line: ', ' '.join(sys.argv), file=stderr)
    stderr.flush()

    mode = sys.argv[1]
    del sys.argv[1]
    cold_parser = parse_tools.cold_parser()
    opts = parse_tools.two_stage_parse(cold_parser)
    # set this to zero if you want to print out a logging header in resume mode as well
    netmisc.set_print_iter(0)

    ae.Metrics(mode, opts).train(0)


if __name__ == '__main__':
    main()

