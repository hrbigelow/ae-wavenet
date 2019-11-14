import sys
from sys import stderr
from pprint import pprint
import torch

import model as ae
import data
import util
import parse_tools  
import checkpoint
import netmisc
import grad_analysis as ga


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
    if opts.hardware == 'CPU':
        opts.device = torch.device('cpu')
    elif opts.hardware == 'GPU':
        if not torch.cuda.is_available():
            raise RuntimeError('GPU requested but not available')
        opts.device = torch.device('cuda')
    elif opts.hardware == 'TPU':
        import torch_xla.core.xla_model as xm
        opts.device = xm.xla_device()
    else:
        raise RuntimeError(
                ('Invalid device {} requested.  ' 
                + 'Must be one of CPU, GPU or TPU').format(opts.hardware))

    print('Using {}'.format(opts.hardware), file=stderr)
    stderr.flush()

    ckpt_path = util.CheckpointPath(opts.ckpt_template)
    learning_rates = dict(zip(opts.learning_rate_steps, opts.learning_rate_rates))

    print('Initializing model and data source...', end='', file=stderr)
    stderr.flush()

    # Construct model
    if mode == 'new':
        pre_params = parse_tools.get_prefixed_items(vars(opts), 'pre_')
        enc_params = parse_tools.get_prefixed_items(vars(opts), 'enc_')
        bn_params = parse_tools.get_prefixed_items(vars(opts), 'bn_')
        dec_params = parse_tools.get_prefixed_items(vars(opts), 'dec_')

        # Initialize data
        data_source = data.Slice(opts.dat_file, opts.n_batch, opts.n_win_batch,
                opts.gpu_resident)

        # Initialize model
        dec_params['n_speakers'] = data_source.num_speakers()

        model = ae.AutoEncoder(pre_params, enc_params, bn_params, dec_params,
                data_source.n_mel_chan, training=True)
        model.encoder.set_parent_vc(data_source.mfcc_vc)

        optim = torch.optim.Adam(params=model.parameters(), lr=learning_rates[0])
        state = checkpoint.State(0, model, data_source, optim)

        state.data.post_init(state.model.decoder.vc)

    else:
        state = checkpoint.State()
        state.load(opts.ckpt_file)
        # print('Restored model, data, and optim from {}'.format(opts.ckpt_file), file=stderr)
        #print('Data state: {}'.format(state.data), file=stderr)
        #print('Model state: {}'.format(state.model.checksum()))
        #print('Optim state: {}'.format(state.optim_checksum()))
        stderr.flush()

    print('Done.', file=stderr)
    stderr.flush()

    start_step = state.step

    # print('Model input size: {}'.format(state.model.input_size), file=stderr)
    # stderr.flush()

    # set this to zero if you want to print out a logging header in resume mode as well
    netmisc.set_print_iter(0)

    state.to(device=opts.device)

    # Initialize optimizer
    metrics = ae.Metrics(state)

    #for p in list(state.model.encoder.parameters()):
    #    with torch.no_grad():
    #        p *= 1 

    # Start training
    print('Training parameters used:', file=stderr)
    pprint(opts, stderr)

    state.init_torch_generator()
    # It doesn't really work to initialize the codebook from data, because
    # the data may produce outlier vectors, and the codebook should not have
    # outlier vectors, since they will dominate if there is a scale mismatch
    if state.model.bn_type in ('vqvae', 'vqvae-ema'):
        state.model.init_codebook(state.data, 10000)

    def update_model_closure():
        metrics.update()
        metrics.loss()

    while state.step < opts.max_steps:
        if state.step in learning_rates:
            state.update_learning_rate(learning_rates[state.step])
        # do 'pip install --upgrade scipy' if you get 'FutureWarning: ...'
        # print('in main loop')

        #if (state.step in (50, 100, 300, 500) and 
        #        state.model.bn_type in ('vqvae', 'vqvae-ema')):
        #    print('Reinitializing embed with current distribution', file=stderr)
        #    stderr.flush()
        #    state.model.init_vq_embed(state.data)

        metrics.update()
        # This is where parameter updates happen

        if state.device.type == 'xla':
            import torch_xla.core.xla_model as xm
            loss = xm.optimizer_step(state.optim, barrier=True,
                    optimizer_args={'closure': metrics.loss})
        else:
            loss = metrics.state.optim.step(metrics.loss)

        if state.model.bn_type == 'vqvae-ema' and state.step > 10000:
            state.model.bottleneck.update_codebook()

        # avg_peak_dist = metrics.peak_dist()
        avg_max = metrics.avg_max()
        avg_prob_target = metrics.avg_prob_target()

        #if state.step % 100 == 0:
        #    qv = ga.grad_stats(state.model, update_model_closure, 50, [0.01, 0.5, 0.99]) 
        #    for par, vals in qv.items():
        #        print('grad_sd_qtiles\t{}\t{}'.format(par, '\t'.join(map('{:5.2f}'.format, vals))))
        #    stderr.flush()


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
            current_stats = {
                    'step': state.step,
                    'loss': loss,
                    'tprb_m': avg_prob_target,
                    # 'pk_d_m': avg_peak_dist
                    }
            #fmt = "M\t{:d}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}"
            #print(fmt.format(state.step, loss, avg_prob_target, avg_peak_dist,
            #    avg_max), file=stderr)
            if state.model.bn_type in ('vqvae', 'vqvae-ema', 'ae'):
                current_stats.update(state.model.objective.metrics)
                
            netmisc.print_metrics(current_stats, 100)
            stderr.flush()
        
        state.step += 1

        # Checkpointing
        if ((state.step % opts.save_interval == 0 and state.step != start_step) or
                (mode == 'new' and state.step == 1)):
            ckpt_file = ckpt_path.path(state.step)
            state.save(ckpt_file)
            print('Saved checkpoint to {}'.format(ckpt_file), file=stderr)
            #print('Optim state: {}'.format(state.optim_checksum()), file=stderr)
            stderr.flush()

if __name__ == '__main__':
    main()

