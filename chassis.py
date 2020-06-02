from sys import stderr
import torch
import data
import autoencoder_model as ae
import mfcc_inverter as mi
import checkpoint
import util
import netmisc
import librosa
import os.path



class GPULoaderIter(object):
    def __init__(self, data_iter):
        self.data_iter = data_iter

    def __iter__(self):
        return self

    def __next__(self):
        return self.data_iter.__next__()[0]


class TPULoaderIter(object):
    def __init__(self, parallel_loader, device):
        self.per_dev_loader = parallel_loader.per_device_loader(device)

    def __iter__(self):
        return self

    def __next__(self):
        vb = self.per_dev_loader.__next__()[0]
        return vb



class Chassis(object):
    """
    Coordinates the construction of the model, dataset, optimizer,
    checkpointing state, and GPU/TPU iterator wrappers.

    Provides a single function for training the model from the constructed
    setup. 

    """

    def __init__(self, mode, opts):
        print('Initializing model and data source...', end='', file=stderr)
        stderr.flush()
        self.learning_rates = dict(zip(opts.learning_rate_steps,
            opts.learning_rate_rates))
        self.opts = opts

        if mode == 'new':
            torch.manual_seed(opts.random_seed)

            # Initialize data
            dataset = data.Slice(opts)
            dataset.load_data(opts.dat_file)
            opts.training = True
            if opts.global_model == 'autoencoder':
                model = ae.AutoEncoder(opts, dataset)
            elif opts.global_model == 'mfcc_inverter':
                model = mi.MfccInverter(opts, dataset)

            model.post_init(dataset)
            dataset.post_init(model)
            optim = torch.optim.Adam(params=model.parameters(), lr=self.learning_rates[0])
            self.state = checkpoint.State(0, model, dataset, optim)
            self.start_step = self.state.step

        else:
            self.state = checkpoint.State()
            self.state.load(opts.ckpt_file, opts.dat_file)
            self.start_step = self.state.step
            # print('Restored model, data, and optim from {}'.format(opts.ckpt_file), file=stderr)
            #print('Data state: {}'.format(state.data), file=stderr)
            #print('Model state: {}'.format(state.model.checksum()))
            #print('Optim state: {}'.format(state.optim_checksum()))
            stderr.flush()

        if self.state.model.bn_type == 'vae':
            self.anneal_schedule = dict(zip(opts.bn_anneal_weight_steps,
                opts.bn_anneal_weight_vals))

        self.ckpt_path = util.CheckpointPath(self.opts.ckpt_template)
        self.quant = None
        self.target = None
        self.softmax = torch.nn.Softmax(1) # input to this is (B, Q, N)

        if self.opts.hwtype == 'GPU':
            self.device = torch.device('cuda')
            self.data_loader = self.state.data_loader
            self.data_loader.set_target_device(self.device)
            self.optim_step_fn = (lambda: self.state.optim.step(self.loss_fn))
            self.data_iter = GPULoaderIter(iter(self.data_loader))
        else:
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.parallel_loader as pl
            self.device = xm.xla_device()
            self.data_loader = pl.ParallelLoader(self.state.data_loader, [self.device])
            self.data_iter = TPULoaderIter(self.data_loader, self.device)
            self.optim_step_fn = (lambda : xm.optimizer_step(self.state.optim,
                    optimizer_args={'closure': self.loss_fn}))

        self.state.init_torch_generator()
        print('Done.', file=stderr)
        stderr.flush()


    def train(self, index):
        ss = self.state 
        ss.to(self.device)
        current_stats = {}

        # for resuming the learning rate 
        sorted_lr_steps = sorted(self.learning_rates.keys())
        lr_index = util.greatest_lower_bound(sorted_lr_steps, ss.step)
        ss.update_learning_rate(self.learning_rates[sorted_lr_steps[lr_index]])

        if ss.model.bn_type != 'none':
            sorted_as_steps = sorted(self.anneal_schedule.keys())
            as_index = util.greatest_lower_bound(sorted_as_steps, ss.step)
            ss.model.objective.update_anneal_weight(self.anneal_schedule[sorted_as_steps[as_index]])

        if ss.model.bn_type in ('vqvae', 'vqvae-ema'):
            ss.model.init_codebook(self.data_iter, 10000)

        while ss.step < self.opts.max_steps:
            if ss.step in self.learning_rates:
                ss.update_learning_rate(self.learning_rates[ss.step])

            if ss.model.bn_type == 'vae' and ss.step in self.anneal_schedule:
                ss.model.objective.update_anneal_weight(self.anneal_schedule[ss.step])

            loss = self.optim_step_fn()

            if ss.model.bn_type == 'vqvae-ema' and ss.step == 10000:
                ss.model.bottleneck.update_codebook()

            if ss.step % self.opts.progress_interval == 0:
                current_stats.update({
                        'step': ss.step,
                        'loss': loss,
                        'lrate': ss.optim.param_groups[0]['lr'],
                        'tprb_m': self.avg_prob_target(),
                        # 'pk_d_m': avg_peak_dist
                        })
                current_stats.update(ss.model.objective.metrics)

                if ss.model.bn_type in ('vae'):
                    current_stats['free_nats'] = ss.model.objective.free_nats
                    current_stats['anneal_weight'] = \
                            ss.model.objective.anneal_weight.item()

                if ss.model.bn_type in ('vqvae', 'vqvae-ema', 'ae', 'vae'):
                    current_stats.update(ss.model.encoder.metrics)

                netmisc.print_metrics(current_stats, index, 100)
                stderr.flush()

            if ((ss.step % self.opts.save_interval == 0 and ss.step !=
                self.start_step)):
                self.save_checkpoint()
            ss.step += 1

    def save_checkpoint(self):
        ckpt_file = self.ckpt_path.path(self.state.step)
        self.state.save(ckpt_file)
        print('Saved checkpoint to {}'.format(ckpt_file), file=stderr)
        #print('Optim state: {}'.format(state.optim_checksum()), file=stderr)
        stderr.flush()

    def run_batch(self):
        """
        run the next batch through the model, populating quantities for the
        loss.
        """
        batch = next(self.data_iter)
        self.quant, self.target, self.loss = self.state.model.run(batch) 
        self.probs = self.softmax(self.quant)
        self.mel_enc_input = batch.mel_enc_input
        

    def loss_fn(self):
        """This is the closure needed for the optimizer"""
        self.run_batch()
        self.state.optim.zero_grad()
        self.loss.backward()
        return self.loss
    
    def peak_dist(self):
        """Average distance between the indices of the peaks in pred and
        target"""
        diffs = torch.argmax(self.quant, dim=1) - self.target.long()
        mean = torch.mean(torch.abs(diffs).float())
        return mean

    def avg_max(self):
        """Average max value for the predictions.  As the prediction becomes
        more peaked, this should go up"""
        max_val, max_ind = torch.max(self.probs, dim=1)
        mean = torch.mean(max_val)
        return mean
        
    def avg_prob_target(self):
        """Average probability given to target"""
        target_probs = torch.gather(self.probs, 1, self.target.long().unsqueeze(1)) 
        mean = torch.mean(target_probs)
        return mean


class InferenceChassis(object):
    """
    Coordinates construction of model and dataset for running inference
    """
    def __init__(self, mode, opts):
        self.state = checkpoint.InferenceState()
        self.state.load(opts.ckpt_file, opts.dat_file)
        self.state.model.eval()
        self.output_dir = opts.output_dir
        self.n_replicas = opts.dec_n_replicas

        if opts.hwtype in ('GPU', 'CPU'):
            if opts.hwtype == 'GPU':
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
            self.data_loader = self.state.data_loader
            self.data_loader.set_target_device(self.device)
            self.data_iter = GPULoaderIter(iter(self.data_loader))
        else:
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.parallel_loader as pl
            self.device = xm.xla_device()
            self.data_loader = pl.ParallelLoader(self.state.data_loader, [self.device])
            self.data_iter = TPULoaderIter(self.data_loader, self.device)

    def infer(self):
        self.state.to(self.device)
        sample_rate = self.state.data_loader.dataset.sample_rate
        n_quant = self.state.model.wavenet.ac.n_quant
        n_rep = torch.tensor(self.n_replicas, device=self.device)

        for mb in self.data_iter:
            out_template = os.path.join(self.output_dir,
                    os.path.basename(os.path.splitext(mb.file_path)[0])
                    + '.{}.wav')

            wav_orig, wav_sample = self.state.model(mb.mel_enc_input,
                    mb.wav_enc_input, mb.voice_index,
                    mb.jitter_index, n_rep)

            # save results to specified files
            for i in range(self.n_replicas):
                wav_final = util.mu_decode_torch(wav_sample[i], n_quant)
                path = out_template.format('rep' + str(i)) 
                librosa.output.write_wav(path, wav_final.cpu().numpy(), sample_rate) 

            wav_final = util.mu_decode_torch(wav_orig, n_quant)
            path = out_template.format('orig') 
            librosa.output.write_wav(path, wav_final.cpu().numpy(), sample_rate) 

            print('Wrote {}'.format(
                out_template.format('0-'+str(n_rep-1))))

