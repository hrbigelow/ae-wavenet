import torch
import wave_encoder as we
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
        opts = parse_tools.two_stage_parse(parse_tools.cold)
    elif mode == 'resume':
        opts = parse_tools.resume.parse_args()  

    opts.device = None
    if not opts.disable_cuda and torch.cuda.is_available():
        opts.device = torch.device('cuda')
    else:
        opts.device = torch.device('cpu') 

    ckpt_path = util.CheckpointPath(opts.ckpt_template)

    # Construct model
    if mode == 'new':
        pre_params = parse_tools.get_prefixed_items(vars(opts), 'pre_')
        enc_params = parse_tools.get_prefixed_items(vars(opts), 'enc_')
        bn_params = parse_tools.get_prefixed_items(vars(opts), 'bn_')
        dec_params = parse_tools.get_prefixed_items(vars(opts), 'dec_')
        sample_catalog = D.parse_sample_catalog(opts.sam_file)

        state = checkpoint.State(0, pre_params, enc_params, bn_params,
                dec_params, sample_catalog, pre_params['sample_rate'],
                opts.frac_permutation_use)
        state.build()

        print('Initializing model parameters', file=stderr)
        state.model.initialize_weights()

    else:
        state = checkpoint.State()
        state.load(opts.ckpt_file)
        print('Restored model and data from {}'.format(opts.ckpt_file), file=stderr)

    state.model.set_geometry(opts.n_sam_per_slice)

    state.data.set_geometry(opts.n_batch, state.model.input_size,
            state.model.output_size, opts.requested_wav_buf_sz)

    state.model.to(device=opts.device)

    #total_bytes = 0
    #for name, par in model.named_parameters():
    #    n_bytes = par.data.nelement() * par.data.element_size()
    #    total_bytes += n_bytes
    #    print(name, type(par.data), par.size(), n_bytes)
    #print('total_bytes: ', total_bytes)

    # Initialize optimizer
    model_params = state.model.parameters()
    loss_fcn = state.model.loss_factory(state.data.batch_slice_gen_fn(), opts.device)

    # Start training
    print('Starting training...', file=stderr)

    learning_rates = dict(zip(opts.learning_rate_steps, opts.learning_rate_rates))
    start_step = state.step
    while state.step < opts.max_steps:
        if state.step in learning_rates:
            optim = torch.optim.Adam(params=model_params,
                    lr=learning_rates[state.step])
        loss = optim.step(loss_fcn)

        if state.step % opts.progress_interval == 0:
            print('Step {}, Loss: {}'.format(state.step, loss))

        if state.step % opts.save_interval == 0 and state.step != start_step:
            ckpt_file = ckpt_path.path(state.step)
            state.save(ckpt_file)
            print('Saved checkpoint to {}'.format(ckpt_file), file=stderr)

        state.step += 1

if __name__ == '__main__':
    main()

