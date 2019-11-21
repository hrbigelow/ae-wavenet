# Full Autoencoder model
from sys import stderr
from hashlib import md5
import numpy as np
from pickle import dumps
import torch
from torch import nn
from torch.nn.modules import loss
from scipy.cluster.vq import kmeans

import model as ae
import checkpoint
import ae_bn
import data
import mfcc
import parse_tools  
import vconv
import util
import netmisc
import vq_bn
import vqema_bn
import vae_bn
import wave_encoder as enc
import wavenet as dec 

# from numpy import vectorize as np_vectorize
class PreProcess(nn.Module):
    """
    Perform one-hot encoding
    """
    def __init__(self, pre_params, n_quant):
        super(PreProcess, self).__init__()
        self.n_quant = n_quant
        self.register_buffer('quant_onehot', torch.eye(self.n_quant))

        # A dummy buffer that simply allows querying the current model device 
        self.register_buffer('dummy_buf', torch.empty(0))

    def one_hot(self, wav_compand):
        """
        wav_compand: (B, T)
        B, Q, T: n_batch, n_quant, n_timesteps
        returns: (B, Q, T)
        """
        wav_compand_tmp = wav_compand.long()
        wav_one_hot = util.gather_md(self.quant_onehot, 0, wav_compand_tmp).permute(1,0,2)
        return wav_one_hot

    def forward(self, in_snd_slice):
        """
        Converts the input to a one-hot format
        """
        in_snd_slice_onehot = self.one_hot(in_snd_slice)
        return in_snd_slice_onehot


class AutoEncoder(nn.Module):
    """
    Full Autoencoder model.  The _initialize method allows us to seamlessly initialize
    from __init__ or __setstate__ 
    """
    def __init__(self, pre_params, enc_params, bn_params, dec_params,
            n_mel_chan, training):
        self.init_args = {
                'pre_params': pre_params,
                'enc_params': enc_params,
                'bn_params': bn_params,
                'dec_params': dec_params,
                'n_mel_chan': n_mel_chan,
                'training': training
                }
        self._initialize()

    def _initialize(self):
        super(AutoEncoder, self).__init__() 
        pre_params = self.init_args['pre_params']
        enc_params = self.init_args['enc_params']
        bn_params = self.init_args['bn_params']
        dec_params = self.init_args['dec_params']
        n_mel_chan = self.init_args['n_mel_chan']
        training = self.init_args['training']

        # the "preprocessing"
        self.preprocess = PreProcess(pre_params, n_quant=dec_params['n_quant'])

        self.encoder = enc.Encoder(n_in=n_mel_chan, parent_vc=None, **enc_params)

        bn_type = bn_params['type']
        bn_extra = dict((k, v) for k, v in bn_params.items() if k != 'type')
    
        # In each case, the objective function's 'forward' method takes the
        # same arguments.
        if bn_type == 'vqvae':
            self.bottleneck = vq_bn.VQ(**bn_extra, n_in=enc_params['n_out'])
            self.objective = vq_bn.VQLoss(self.bottleneck)

        elif bn_type == 'vqvae-ema':
            self.bottleneck = vqema_bn.VQEMA(**bn_extra, n_in=enc_params['n_out'],
                    training=training)
            self.objective = vqema_bn.VQEMALoss(self.bottleneck)

        elif bn_type == 'vae':
            # mu and sigma members  
            self.bottleneck = vae_bn.VAE(**bn_extra, n_in=enc_params['n_out'])
            self.objective = vae_bn.SGVBLoss(self.bottleneck)

        elif bn_type == 'ae':
            self.bottleneck = ae_bn.AE(n_out=bn_extra['n_out'], n_in=enc_params['n_out'])
            self.objective = ae_bn.AELoss(self.bottleneck, 0.001) 

        else:
            raise InvalidArgument('bn_type must be one of "ae", "vae", or "vqvae"')

        self.bn_type = bn_type
        self.decoder = dec.WaveNet(
                **dec_params,
                parent_vc=self.encoder.vc,
                n_lc_in=bn_params['n_out']
                )
        self.vc = self.decoder.vc
        self.decoder.post_init()


    def __getstate__(self):
        state = { 
                'init_args': self.init_args,
                'state_dict': self.state_dict()
                }
        return state 

    def __setstate__(self, state):
        self.init_args = state['init_args']
        self._initialize()
        self.load_state_dict(state['state_dict'])


    def init_codebook(self, data_source, n_samples):
        """
        Initialize the VQ Embedding with samples from the encoder
        """
        if self.bn_type not in ('vqvae', 'vqvae-ema'):
            raise RuntimeError('init_vq_embed only applies to the vqvae model type')

        bn = self.bottleneck
        e = 0
        n_codes = bn.emb.shape[0]
        k = bn.emb.shape[1]
        samples = np.empty((n_samples, k), dtype=np.float) 
        
        with torch.no_grad():
            while e != n_samples:
                vbatch = next(data_source)
                encoding = self.encoder(vbatch.mel_input)
                ze = self.bottleneck.linear(encoding)
                ze = ze.permute(0, 2, 1).flatten(0, 1)
                c = min(n_samples - e, ze.shape[0])
                samples[e:e + c,:] = ze.cpu()[0:c,:]
                e += c

        km, __ = kmeans(samples, n_codes)
        bn.emb[...] = torch.from_numpy(km)

        if self.bn_type == 'vqvae-ema':
            bn.ema_numer = bn.emb * bn.ema_gamma_comp
            bn.ema_denom = bn.n_sum_ones * bn.ema_gamma_comp
        
    def checksum(self):
        """Return checksum of entire set of model parameters"""
        return util.tensor_digest(self.parameters())
        

    def forward(self, mels, wav_onehot_dec, voice_inds, lcond_slice):
        """
        B: n_batch
        M: n_mels
        T: receptive field of autoencoder
        T': receptive field of decoder 
        R: size of local conditioning output of encoder (T - encoder.vc.total())
        N: n_win (# consecutive samples processed in one batch channel)
        Q: n_quant
        mels: (B, M, T)
        wav_compand: (B, T)
        wav_onehot_dec: (B, T')  
        Outputs: 
        quant_pred (B, Q, N) # predicted wav amplitudes
        """
        encoding = self.encoder(mels)
        encoding_bn = self.bottleneck(encoding)
        self.encoding_bn = encoding_bn
        quant = self.decoder(wav_onehot_dec, encoding_bn, voice_inds,
                lcond_slice)
        return quant

    def run(self, vbatch):
        """
        Run the model on one batch, returning the predicted and
        actual output
        B, T, Q: n_batch, n_timesteps, n_quant
        Outputs:
        quant_pred: (B, Q, T) (the prediction from the model)
        wav_batch_out: (B, T) (the actual data from the same timesteps)
        """
        wav_onehot_dec = self.preprocess(vbatch.wav_input)
        # grad = torch.autograd.grad(wav_onehot_dec, vbatch.wav_input).data

        # Slice each wav input
        wav_batch_out = vbatch.wav_input.new_empty(vbatch.batch_size,
                vbatch.loss_wav_len()) 
        for b, (sl_b, sl_e) in enumerate(vbatch.loss_wav_slice):
            wav_batch_out[b] = vbatch.wav_input[b,sl_b:sl_e]

        # self.wav_batch_out = wav_batch_out
        self.wav_onehot_dec = wav_onehot_dec

        quant = self.forward(vbatch.mel_input, wav_onehot_dec,
                vbatch.voice_index, vbatch.lcond_slice)
        # quant_pred[:,:,0] is a prediction for wav_compand_out[:,1] 
        return quant[...,:-1], wav_batch_out[...,1:]


class GPULoaderIter(object):
    def __init__(self, data_iter):
        self.data_iter = data_iter

    def __next__(self):
        return self.data_iter.__next__()[0]


class TPULoaderIter(object):
    def __init__(self, parallel_loader, device):
        self.per_dev_loader = parallel_loader.per_device_loader(device)

    def __next__(self):
        return self.per_dev_loader.__next__()[0]




class Metrics(object):
    """
    Manage running the model and saving output and target state
    """

    def __init__(self, mode, opts):
        print('Initializing model and data source...', end='', file=stderr)
        stderr.flush()
        self.learning_rates = dict(zip(opts.learning_rate_steps,
            opts.learning_rate_rates))
        self.opts = opts

        if mode == 'new':
            pre_params = parse_tools.get_prefixed_items(vars(opts), 'pre_')
            enc_params = parse_tools.get_prefixed_items(vars(opts), 'enc_')
            bn_params = parse_tools.get_prefixed_items(vars(opts), 'bn_')
            dec_params = parse_tools.get_prefixed_items(vars(opts), 'dec_')

            # Initialize data
            dataset = data.Slice(opts.dat_file, opts.n_batch, opts.n_win_batch)
            dec_params['n_speakers'] = dataset.num_speakers()
            model = ae.AutoEncoder(pre_params, enc_params, bn_params, dec_params,
                    dataset.n_mel_chan, training=True)
            model.encoder.set_parent_vc(dataset.mfcc_vc)
            dataset.post_init(model.decoder.vc)
            optim = torch.optim.Adam(params=model.parameters(), lr=self.learning_rates[0])
            self.state = checkpoint.State(0, model, dataset, optim)
            self.start_step = self.state.step

        else:
            self.state = checkpoint.State()
            self.state.load(opts.ckpt_file)
            self.start_step = self.state.step
            # print('Restored model, data, and optim from {}'.format(opts.ckpt_file), file=stderr)
            #print('Data state: {}'.format(state.data), file=stderr)
            #print('Model state: {}'.format(state.model.checksum()))
            #print('Optim state: {}'.format(state.optim_checksum()))
            stderr.flush()

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
        if ss.model.bn_type in ('vqvae', 'vqvae-ema'):
            ss.model.init_codebook(self.data_iter, 10000)

        while ss.step < self.opts.max_steps:
            if ss.step in self.learning_rates:
                ss.update_learning_rate(self.learning_rates[ss.step])
            loss = self.update()
            if ss.model.bn_type == 'vqvae-ema' and ss.step == 10000:
                ss.model.bottleneck.update_codebook()

            if ss.step % self.opts.progress_interval == 0:
                current_stats = {
                        'step': ss.step,
                        'loss': loss,
                        'tprb_m': self.avg_prob_target(),
                        # 'pk_d_m': avg_peak_dist
                        }
                if ss.model.bn_type in ('vqvae', 'vqvae-ema', 'ae'):
                    current_stats.update(ss.model.objective.metrics)
                netmisc.print_metrics(current_stats, 100)
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

    def update(self):
        batch = next(self.data_iter)
        quant_pred_snip, wav_compand_out_snip = self.state.model.run(batch) 
        self.quant = quant_pred_snip
        self.target = wav_compand_out_snip
        self.probs = self.softmax(self.quant)
        self.mel_input = batch.mel_input
        return self.optim_step_fn()
        

    def loss_fn(self):
        """This is the closure needed for the optimizer"""
        if self.quant is None or self.target is None:
            raise RuntimeError('Must call update() first')
        self.state.optim.zero_grad()
        loss = self.state.model.objective(self.quant, self.target)
        inputs = (self.mel_input, self.state.model.encoding_bn)
        mel_grad, bn_grad = torch.autograd.grad(loss, inputs, retain_graph=True)
        # print(mel_grad)
        # print(bn_grad)
        self.state.model.objective.metrics.update({
            'mel_grad_sd': mel_grad.std(),
            # 'mel_grad_max': mel_grad.max(),
            'bn_grad_sd': bn_grad.std(),
            # 'bn_grad_max': bn_grad.max()
            })
        # loss.backward(create_graph=True, retain_graph=True)
        loss.backward()
        return loss
    
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



#    def train_old(self):
#        """
#        Run the main training logic
#        """
#        ss = self.state
#        ss.to(device=ss.device)
#        self.data_loader = ss.data_loader
#
#        while ss.step < self.max_steps:
#            if ss.step in learning_rates:
#                ss.update_learning_rate(learning_rates[ss.step])
#            # do 'pip install --upgrade scipy' if you get 'FutureWarning: ...'
#            # print('in main loop')
#
#            #if (ss.step in (50, 100, 300, 500) and 
#            #        ss.model.bn_type in ('vqvae', 'vqvae-ema')):
#            #    print('Reinitializing embed with current distribution', file=stderr)
#            #    stderr.flush()
#            #    ss.model.init_vq_embed(ss.data)
#
#            loss = metrics.update()
#            if ss.model.bn_type == 'vqvae-ema' and ss.step > 10000:
#                ss.model.bottleneck.update_codebook()
#
#            # avg_peak_dist = metrics.peak_dist()
#            avg_max = self.avg_max()
#            avg_prob_target = self.avg_prob_target()
#
#            if False:
#                for n, p in list(ss.model.encoder.named_parameters()):
#                    g = p.grad
#                    if g is None:
#                        print('{:60s}\tNone'.format(n), file=stderr)
#                    else:
#                        fmt='{:s}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'
#                        print(fmt.format(n, g.max(), g.min(), g.mean(), g.std()), file=stderr)
#
#            # Progress reporting
#            if ss.step % opts.progress_interval == 0:
#                current_stats = {
#                        'step': ss.step,
#                        'loss': loss,
#                        'tprb_m': avg_prob_target,
#                        # 'pk_d_m': avg_peak_dist
#                        }
#                #fmt = "M\t{:d}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}"
#                #print(fmt.format(ss.step, loss, avg_prob_target, avg_peak_dist,
#                #    avg_max), file=stderr)
#                if ss.model.bn_type in ('vqvae', 'vqvae-ema', 'ae'):
#                    current_stats.update(ss.model.objective.metrics)
#                    
#                netmisc.print_metrics(current_stats, 100)
#                stderr.flush()
#            
#            # Checkpointing
#            if ((ss.step % self.opts.save_interval == 0 and ss.step !=
#                self.start_step)):
#                self.save_checkpoint()
#
#            ss.step += 1
#
