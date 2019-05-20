import torch
import model as ae
import data as D 
import util
import parse_tools  
import sys
import checkpoint
import netmisc
from sys import stderr

def main():
    if len(sys.argv) == 1 or sys.argv[1] not in ('new', 'resume'):
        print(parse_tools.top_usage, file=stderr)
        return
    print('Command line: ', ' '.join(sys.argv), file=stderr)
    stderr.flush()

    mode = sys.argv[1]
    del sys.argv[1]
    if mode == 'new':
        cold_parser = parse_tools.cold_parser()
        opts = parse_tools.two_stage_parse(cold_parser)
    elif mode == 'resume':
        resume_parser = parse_tools.resume_parser()
        opts = resume_parser.parse_args()  

    opts.device = None
    if not opts.disable_cuda and torch.cuda.is_available():
        opts.device = torch.device('cuda')
        print('Using GPU', file=stderr)
    else:
        opts.device = torch.device('cpu') 
        print('Using CPU', file=stderr)
    stderr.flush()

    ckpt_path = util.CheckpointPath(opts.ckpt_template)
    learning_rates = dict(zip(opts.learning_rate_steps, opts.learning_rate_rates))

    # Construct model
    if mode == 'new':
        # Initialize model
        pre_params = parse_tools.get_prefixed_items(vars(opts), 'pre_')
        enc_params = parse_tools.get_prefixed_items(vars(opts), 'enc_')
        bn_params = parse_tools.get_prefixed_items(vars(opts), 'bn_')
        dec_params = parse_tools.get_prefixed_items(vars(opts), 'dec_')

        # Initialize data
        sample_catalog = D.parse_sample_catalog(opts.sam_file)
        data = D.WavSlices(sample_catalog, pre_params['sample_rate'],
                opts.frac_permutation_use, opts.requested_wav_buf_sz)
        dec_params['n_speakers'] = data.num_speakers()

        model = ae.AutoEncoder(pre_params, enc_params, bn_params, dec_params,
                opts.n_sam_per_slice)
        optim = torch.optim.Adam(params=model.parameters(), lr=learning_rates[0])
        state = checkpoint.State(0, model, data, optim)

    else:
        state = checkpoint.State()
        state.load(opts.ckpt_file)
        state.model.set_slice_size(opts.n_sam_per_slice)
        print('Restored model, data, and optim from {}'.format(opts.ckpt_file), file=stderr)
        #print('Data state: {}'.format(state.data), file=stderr)
        #print('Model state: {}'.format(state.model.checksum()))
        #print('Optim state: {}'.format(state.optim_checksum()))
        stderr.flush()

    start_step = state.step
    netmisc.set_print_iter(start_step)

    state.data.set_geometry(opts.n_batch, state.model.input_size,
            state.model.output_size)
    state.to(device=opts.device)

    # Initialize optimizer
    metrics = ae.Metrics(state)
    batch_gen = state.data.batch_slice_gen_fn()

    #for p in list(state.model.encoder.parameters()):
    #    with torch.no_grad():
    #        p *= 1 

    # Start training
    print('Training parameters used:', opts, file=stderr)
    hdr_fmt = 'M\t{:s}\t{:s}\t{:s}\t{:s}\t{:s}'
    print(hdr_fmt.format('Step', 'Loss', 'AvProbTrg', 'PeakDist', 'AvgMax'), file=stderr)
    stderr.flush()

    state.init_torch_generator()
    #print('Generator state: {}'.format(util.tensor_digest(torch.get_rng_state())))

    while state.step < opts.max_steps:
        #if state.step in learning_rates:
        #    state.update_learning_rate(learning_rates[state.step])
        # do 'pip install --upgrade scipy' if you get 'FutureWarning: ...'
        metrics.update(batch_gen)
        loss = metrics.state.optim.step(metrics.loss)
        avg_peak_dist = metrics.peak_dist()
        avg_max = metrics.avg_max()
        avg_prob_target = metrics.avg_prob_target()

        if state.step % 500 == 5 and state.step < 3000 and state.model.bn_type == 'vqvae':
            print('Reinitializing embed with current distribution', file=stderr)
            stderr.flush()
            state.model.init_vq_embed(batch_gen)

        if False:
            for n, p in list(state.model.encoder.named_parameters()):
                g = p.grad
                if g is None:
                    print('{:60s}\tNone'.format(n), file=stderr)
                else:
                    fmt='{:s}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'
                    print(fmt.format(n, g.max(), g.min(), g.mean(), g.std()), file=stderr)

        # Progress reporting
        if state.step % opts.progress_interval == 0:
            fmt = "M\t{:d}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}"
            print(fmt.format(state.step, loss, avg_prob_target, avg_peak_dist,
                avg_max), file=stderr)
            stderr.flush()
        
        state.step += 1

        # Checkpointing
        if state.step % opts.save_interval == 0 and state.step != start_step:
            ckpt_file = ckpt_path.path(state.step)
            state.save(ckpt_file)
            print('Saved checkpoint to {}'.format(ckpt_file), file=stderr)
            #print('Model state: {}'.format(state.model.checksum()), file=stderr)
            #print('Generator state: {}'.format(util.tensor_digest(torch.get_rng_state())),
            #        file=stderr)
            #print('Optim state: {}'.format(state.optim_checksum()), file=stderr)
            # print('Data position: ', state.data, file=stderr)
            stderr.flush()


if __name__ == '__main__':
    main()

