import torch
import model as ae
import data as D 
import util
import parse_tools  
import sys
import checkpoint
from sys import stderr

def main():
    if len(sys.argv) == 1 or sys.argv[1] not in ('new', 'resume'):
        print(parse_tools.top_usage, file=stderr)
        return

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
    else:
        opts.device = torch.device('cpu') 

    ckpt_path = util.CheckpointPath(opts.ckpt_template)

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

        #with torch.autograd.set_detect_anomaly(True):
        model = ae.AutoEncoder(pre_params, enc_params, bn_params, dec_params,
                opts.n_sam_per_slice)

        # Construct overall state
        state = checkpoint.State(0, model, data)

    else:
        state = checkpoint.State()
        state.load(opts.ckpt_file)
        state.model.set_slice_size(opts.n_sam_per_slice)
        print('Restored model and data from {}'.format(opts.ckpt_file), file=stderr)


    state.data.set_geometry(opts.n_batch, state.model.input_size,
            state.model.output_size)

    state.model.to(device=opts.device)

    #total_bytes = 0
    #for name, par in model.named_parameters():
    #    n_bytes = par.data.nelement() * par.data.element_size()
    #    total_bytes += n_bytes
    #    print(name, type(par.data), par.size(), n_bytes)
    #print('total_bytes: ', total_bytes)

    # Initialize optimizer
    model_params = state.model.parameters()
    metrics = ae.Metrics(state.model, None)
    batch_gen = state.data.batch_slice_gen_fn()

    #loss_fcn = state.model.loss_factory(state.data.batch_slice_gen_fn())

    # Start training
    print('Starting training...', file=stderr)
    print("Step\tLoss\tAvProbTrg\tPeakDist\tAvgMax", file=stderr)
    stderr.flush()

    learning_rates = dict(zip(opts.learning_rate_steps, opts.learning_rate_rates))
    start_step = state.step
    if start_step not in learning_rates:
        ref_step = util.greatest_lower_bound(opts.learning_rate_steps, start_step)
        metrics.optim = torch.optim.Adam(params=model_params,
                lr=learning_rates[ref_step])

    while state.step < opts.max_steps:
        if state.step in learning_rates:
            metrics.optim = torch.optim.Adam(params=model_params,
                    lr=learning_rates[state.step])
        # do 'pip install --upgrade scipy' if you get 'FutureWarning: ...'
        metrics.update(batch_gen)
        loss = metrics.optim.step(metrics.loss)
        avg_peak_dist = metrics.peak_dist()
        avg_max = metrics.avg_max()
        avg_prob_target = metrics.avg_prob_target()

        # Progress reporting
        if state.step % opts.progress_interval == 0:
            fmt = "{}\t{:10.5f}\t{:10.5f}\t{:10.5f}\t{:10.5f}"
            print(fmt.format(state.step, loss, avg_prob_target, avg_peak_dist,
                avg_max), file=stderr)
            stderr.flush()

        # Checkpointing
        if state.step % opts.save_interval == 0 and state.step != start_step:
            ckpt_file = ckpt_path.path(state.step)
            state.save(ckpt_file)
            print('Saved checkpoint to {}'.format(ckpt_file), file=stderr)

        state.step += 1

if __name__ == '__main__':
    main()

