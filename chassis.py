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
    def __init__(self, loader, device):
        self.loader_iter = iter(loader)
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        items = next(self.loader_iter)
        return tuple(item.to(self.device) for item in items)


def reduce_add(vlist):
    return torch.stack(vlist).sum(dim=0)

def reduce_mean(vlist):
    return torch.stack(vlist).mean(dim=0)

class Chassis(object):
    """
    Coordinates the construction of the model, dataset, optimizer,
    checkpointing state, and GPU/TPU iterator wrappers.

    Provides a single function for training the model from the constructed
    setup. 

    """
    def __init__(self, hps, dat_file):
        if hps.hw in ('TPU', 'TPU-single'):
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.parallel_loader as pl
            num_replicas = xm.xrt_world_size()
            rank = xm.get_ordinal()
        else:
            num_replicas = 1
            rank = 0

        if hps.hw not in ('TPU', 'TPU-single') or xm.is_master_ordinal():
            print('Initializing model and data source...', end='', file=stderr)
            stderr.flush()

        self.state = checkpoint.State(hps, dat_file, train_mode=True,
                ckpt_file=hps.get('ckpt_file', None), step=0,
                num_replicas=num_replicas, rank=rank)

        self.learning_rates = dict(zip(hps.learning_rate_steps,
            hps.learning_rate_rates))

        if self.state.model.bn_type == 'vae':
            self.anneal_schedule = dict(zip(hps.bn_anneal_weight_steps,
                hps.bn_anneal_weight_vals))

        if hps.hw not in ('TPU', 'TPU-single') or xm.is_master_ordinal():
            self.ckpt_path = util.CheckpointPath(hps.ckpt_template)

        self.softmax = torch.nn.Softmax(1) # input to this is (B, Q, N)
        self.hw = hps.hw

        if hps.hw == 'GPU':
            device = torch.device('cuda')
            self.device_loader = GPULoaderIter(self.state.data.loader, device)
            self.state.to(device)
        else:
            device = xm.xla_device()
            para_loader = pl.ParallelLoader(self.state.data.loader, [device])
            self.device_loader = para_loader.per_device_loader(device) 
            self.num_devices = xm.xrt_world_size()
            self.state.to(device)

        self.state.init_torch_generator()

        if hps.hw not in ('TPU', 'TPU-single') or xm.is_master_ordinal():
            print('Done.', file=stderr)
            stderr.flush()


    def train(self, hps, index):
        is_tpu = (hps.hw in ('TPU', 'TPU-single'))
        if is_tpu:
            import torch_xla.core.xla_model as xm

        ss = self.state 
        current_stats = {}

        # for resuming the learning rate 
        sorted_lr_steps = sorted(self.learning_rates.keys())
        lr_index = util.greatest_lower_bound(sorted_lr_steps, ss.data.global_step)
        ss.update_learning_rate(self.learning_rates[sorted_lr_steps[lr_index]])

        if ss.model.bn_type != 'none':
            sorted_as_steps = sorted(self.anneal_schedule.keys())
            as_index = util.greatest_lower_bound(sorted_as_steps,
                    ss.data.global_step)
            ss.model.objective.update_anneal_weight(self.anneal_schedule[sorted_as_steps[as_index]])

        if ss.model.bn_type in ('vqvae', 'vqvae-ema'):
            ss.model.init_codebook(self.data_iter, 10000)
        
        for batch_num, batch in enumerate(self.device_loader):
            wav, mel, voice, jitter, position = batch

            if ss.data.global_step in self.learning_rates:
                ss.update_learning_rate(self.learning_rates[ss.data.global_step])

            if ss.model.bn_type == 'vae' and ss.step in self.anneal_schedule:
                ss.model.objective.update_anneal_weight(self.anneal_schedule[ss.data.global_step])

            ss.optim.zero_grad()
            quant, self.target, loss = self.state.model.run(wav, mel, voice, jitter) 
            self.probs = self.softmax(quant)
            self.mel_enc_input = mel
            loss.backward()

            if is_tpu:
                xm.optimizer_step(ss.optim)
            else:
                ss.optim.step()

            if ss.model.bn_type == 'vqvae-ema' and ss.data.global_step == 10000:
                ss.model.bottleneck.update_codebook()

            tprb_m = self.avg_prob_target()

            if batch_num % hps.progress_interval == 0:
                if is_tpu:
                    loss_red = xm.mesh_reduce('mesh_loss', loss, reduce_mean)
                    tprb_m_red = xm.mesh_reduce('mesh_tprb_m', tprb_m, reduce_mean)
                    # print(f'index: {index}, loss: {loss}, loss_reduced: {loss_reduced}',
                    #         file=stderr)
                else:
                    loss_red = loss
                    tprb_m_red = tprb_m

                current_stats.update({
                        'gstep': ss.data.global_step,
                        'epoch': position[0],
                        'step': position[1],
                        'loss': loss,
                        'loss_r': loss_red,
                        'lrate': ss.optim.param_groups[0]['lr'],
                        'tprb_m': tprb_m,
                        'tprb_m_r': tprb_m_red
                        # 'pk_d_m': avg_peak_dist
                        })
                current_stats.update(ss.model.objective.metrics)

                if ss.model.bn_type in ('vae'):
                    current_stats['free_nats'] = ss.model.objective.free_nats
                    current_stats['anneal_weight'] = \
                            ss.model.objective.anneal_weight.item()

                if ss.model.bn_type in ('vqvae', 'vqvae-ema', 'ae', 'vae'):
                    current_stats.update(ss.model.encoder.metrics)

                # if not is_tpu or xm.is_master_ordinal():
                if True:
                    netmisc.print_metrics(current_stats, index, 100)
                    stderr.flush()

            if (batch_num % hps.save_interval == 0 and batch_num != 0):
                if not is_tpu or xm.is_master_ordinal():
                    self.save_checkpoint()

    def save_checkpoint(self):
        ckpt_file = self.ckpt_path.path(self.state.data.step)
        self.state.save(ckpt_file)
        print('Saved checkpoint to {}'.format(ckpt_file), file=stderr)
        #print('Optim state: {}'.format(state.optim_checksum()), file=stderr)
        stderr.flush()

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


class DataContainer(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])

    def forward(self):
        pass


class InferenceChassis(object):
    """
    Coordinates construction of model and dataset for running inference
    """
    def __init__(self, mode, opts):
        self.output_dir = opts.output_dir
        self.n_replicas = opts.dec_n_replicas
        self.data_write_tmpl = opts.data_write_tmpl

        self.state = checkpoint.InferenceState()
        self.state.load(opts.ckpt_file, opts.dat_file)
        self.state.model.wavenet.set_n_replicas(self.n_replicas)
        self.state.model.eval()

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

    def infer(self, model_scr=None):
        self.state.to(self.device)
        sample_rate = self.state.data_loader.dataset.sample_rate
        n_quant = self.state.model.wavenet.n_quant

        for vb in self.data_iter:
            if self.data_write_tmpl:
                dc = torch.jit.script(DataContainer({
                    'mel': vb.mel,
                    'wav': vb.wav,
                    'voice': vb.voice_idx,
                    'jitter': vb.jitter_idx
                    }))
                dc.save(self.data_write_tmpl)
                print('saved {}'.format(self.data_write_tmpl))

            out_template = os.path.join(self.output_dir,
                    os.path.basename(os.path.splitext(vb.file_path)[0])
                    + '.{}.wav')

            if model_scr:
                with torch.no_grad():
                    wav = model_scr(vb)
            else:
                wav = self.state.model(vb)

            wav_orig, wav_sample = wav[0,...], wav[1:,...]

            # save results to specified files
            for i in range(self.n_replicas):
                wav_final = util.mu_decode_torch(wav_sample[i], n_quant)
                path = out_template.format('rep' + str(i)) 
                librosa.output.write_wav(path, wav_final.cpu().numpy(), sample_rate) 

            wav_final = util.mu_decode_torch(wav_orig, n_quant)
            path = out_template.format('orig') 
            librosa.output.write_wav(path, wav_final.cpu().numpy(), sample_rate) 

            print('Wrote {}'.format(
                out_template.format('0-'+str(self.n_replicas-1))))

