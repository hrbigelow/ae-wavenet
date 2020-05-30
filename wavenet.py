import torch
from torch import nn
from torch.distributions import one_hot_categorical as dcat
import vconv
import numpy as np
from numpy import prod as np_prod
import util
import netmisc

# from numpy import vectorize as np_vectorize
class PreProcess(nn.Module):
    """
    Perform one-hot encoding
    """
    def __init__(self, n_quant):
        super(PreProcess, self).__init__()
        self.n_quant = n_quant
        self.register_buffer('quant_onehot', torch.eye(self.n_quant))

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




class GatedResidualCondConv(nn.Module):
    def __init__(self, wavenet_vc, n_cond, n_res, n_dil, n_skp, stride, dil,
            filter_sz=2, bias=True, parent_vc=None, name=None):
        """
        filter_sz: # elements in the dilated kernels
        n_cond: # channels of local condition vectors
        n_res : # residual channels
        n_dil : # output channels for dilated kernel
        n_skp : # channels output to skip connections
        """
        super(GatedResidualCondConv, self).__init__()
        self.wavenet_vc = wavenet_vc 
        self.conv_signal = nn.Conv1d(n_res, n_dil, filter_sz, dilation=dil, bias=bias)
        self.conv_gate = nn.Conv1d(n_res, n_dil, filter_sz, dilation=dil, bias=bias)
        self.proj_signal = nn.Conv1d(n_cond, n_dil, kernel_size=1, bias=False)
        self.proj_gate = nn.Conv1d(n_cond, n_dil, kernel_size=1, bias=False)
        self.dil_res = nn.Conv1d(n_dil, n_res, kernel_size=1, bias=False)
        self.dil_skp = nn.Conv1d(n_dil, n_skp, kernel_size=1, bias=False)

        # The dilated autoregressive convolution produces an output at the
        # right-most position of the receptive field.  (At the very end of a
        # stack of these, the output corresponds to the position just after
        # this, but within the stack of convolutions, outputs right-aligned.
        dil_filter_sz = (filter_sz - 1) * dil + 1
        self.vc = vconv.VirtualConv(filter_info=(dil_filter_sz - 1, 0),
                parent=parent_vc, name=name)
        self.apply(netmisc.xavier_init)

    def post_init(self):
        """
        Initialize offset tensors
        """
        self.register_buffer('cond_lead', torch.empty(1, dtype=torch.int32))
        self.register_buffer('skip_lead', torch.empty(1, dtype=torch.int32))
        self.register_buffer('left_wing_size',
                torch.tensor([self.vc.l_wing_sz], dtype=torch.int32))
        self.update_leads()

    def update_leads(self):
        """
        Update skip_lead and cond_lead to reflect changed geometry
        or chunk size.  Call this after vconv.compute_inputs is called
        """
        cond_lead, r_off = vconv.output_offsets(self.wavenet_vc['beg_grcc'],
                self.vc)
        assert r_off == 0
        self.cond_lead[0] = cond_lead

        if self.vc == self.wavenet_vc['end_grcc']:
            skip_lead = 0
        else:
            skip_lead, r_off = vconv.output_offsets(self.vc.child,
                    self.wavenet_vc['end_grcc'])
            assert r_off == 0

        self.skip_lead[0] = skip_lead

    def set_incremental(self):
        """
        Set skip_lead and cond_lead for incremental operation
        """
        self.skip_lead[0] = 0
        self.cond_lead[0] = 0 

    def forward(self, x, cond):
        """
        B, T: batchsize, win_size (determined from input)
        C, R, D, S: n_cond, n_res, n_dil, n_skp
        x: (B, R, T) (necessary shape for Conv1d)
        cond: (B, C, T) (necessary shape for Conv1d)
        returns: sig: (B, R, T), skp: (B, S, T) 
        """
        filt = self.conv_signal(x) + \
                self.proj_signal(cond[:,:,self.cond_lead[0]:])
        gate = self.conv_gate(x) + self.proj_gate(cond[:,:,self.cond_lead[0]:])
        z = torch.tanh(filt) * torch.sigmoid(gate)
        sig = self.dil_res(z)
        skp = self.dil_skp(z[:,:,self.skip_lead[0]:])
        sig += x[:,:,self.left_wing_size[0]:]

        return sig, skp 


class Conditioning(nn.Module):
    """
    Module for merging up-sampled local conditioning vectors
    with voice ids.
    """
    def __init__(self, n_speakers, n_embed, bias=True):
        super(Conditioning, self).__init__()
        # Look at nn.embedding
        self.speaker_embedding = nn.Linear(n_speakers, n_embed, bias)
        self.register_buffer('eye', torch.eye(n_speakers))
        self.apply(netmisc.xavier_init)

    def forward(self, lc, speaker_inds):
        """
        I, G, S: n_in_chan, n_embed_chan, n_speakers
        lc : (B, T, I)
        speaker_inds: (B)
        returns: (B, T, I+G)
        """
        assert speaker_inds.dtype == torch.long
        # one_hot: (B, S)
        one_hot = util.gather_md(self.eye, 0, speaker_inds).permute(1, 0) 
        gc = self.speaker_embedding(one_hot) # gc: (B, G)
        gc_rep = gc.unsqueeze(2).expand(-1, -1, lc.shape[2])
        all_cond = torch.cat((lc, gc_rep), dim=1) 
        return all_cond

class Upsampling(nn.Module):
    def __init__(self, n_chan, filter_sz, stride, parent_vc, bias=True, name=None):
        super(Upsampling, self).__init__()
        # See upsampling_notes.txt: padding = filter_sz - stride
        # and: left_offset = left_wing_sz - end_padding
        end_padding = stride - 1
        self.vc = vconv.VirtualConv(
                filter_info=filter_sz, stride=stride,
                padding=(end_padding, end_padding), is_downsample=False,
                parent=parent_vc, name=name
                )

        self.tconv = nn.ConvTranspose1d(n_chan, n_chan, filter_sz, stride,
                padding=filter_sz - stride, bias=bias)
        self.apply(netmisc.xavier_init)

    def forward(self, lc):
        """
        B, T, S, C: batch_sz, timestep, less-frequent timesteps, input channels
        lc: (B, C, S)
        returns: (B, C, T)
        """
        lc_up = self.tconv(lc)
        return lc_up

class Conv1dWrap(nn.Conv1d):
    """
    Simple wrapper that ensures initialization
    """
    def __init__(self, name, parent_vc, **kwargs):
        super(Conv1dWrap, self).__init__(**kwargs)
        self.apply(netmisc.xavier_init)
        self.vc = vconv.VirtualConv(filter_info=kwargs['kernel_size'],
                stride=kwargs['stride'],
                name=name, parent=parent_vc)


class ArchConfig(object):
    def __init__(self, n_lc_in, n_lc_out, n_res, n_dil, n_skp, n_post, n_quant,
            n_blocks, n_block_layers, n_speakers, n_global_embed):
        self.n_lc_in = n_lc_in
        self.n_lc_out = n_lc_out
        self.n_res = n_res
        self.n_dil = n_dil
        self.n_skp = n_skp
        self.n_post = n_post
        self.n_quant = n_quant
        self.n_blocks = n_blocks
        self.n_block_layers = n_block_layers
        self.n_speakers = n_speakers,
        self.n_global_embed = n_global_embed



class WaveNet(nn.Module):
    def __init__(self, filter_sz, n_lc_in, n_lc_out, lc_upsample_filt_sizes,
            lc_upsample_strides, n_res, n_dil, n_skp, n_post, n_quant,
            n_blocks, n_block_layers, n_speakers, n_global_embed,
            bias=True, parent_vc=None):
        super(WaveNet, self).__init__()

        self.ac = ArchConfig(n_lc_in, n_lc_out, n_res, n_dil, n_skp, n_post,
                n_quant, n_blocks, n_block_layers, n_speakers, n_global_embed)

        self.n_blocks = n_blocks
        self.n_block_layers = n_block_layers
        self.n_quant = n_quant
        self.quant_onehot = None 
        self.bias = bias
        post_jitter_filt_sz = 3
        lc_input_stepsize = np_prod(lc_upsample_strides) 

        lc_conv_name = 'LC_Conv(filter_size={})'.format(post_jitter_filt_sz) 
        self.lc_conv = Conv1dWrap(lc_conv_name, parent_vc, in_channels=n_lc_in,
                out_channels=n_lc_out, kernel_size=post_jitter_filt_sz,
                stride=1, bias=self.bias)

        # cur_vc = vconv.VirtualConv(filter_info=post_jitter_filt_sz,
        #         stride=1, parent=parent_vc, name=lc_conv_name)

        self.vc = dict()
        self.vc['beg'] = self.lc_conv.vc 
        cur_vc = self.vc['beg']
        
        # This VC is the first processing of the local conditioning after the
        # Jitter. It is the starting point for the commitment loss aggregation
        self.lc_upsample = nn.Sequential()

        # WaveNet is a stand-alone model, so parent_vc is None
        # The Autoencoder model in model.py will link parent_vcs together.
        for i, (filt_sz, stride) in enumerate(zip(lc_upsample_filt_sizes,
            lc_upsample_strides)):
            name = 'Upsampling_{}(filter_sz={}, stride={})'.format(i, filt_sz, stride)   
            mod = Upsampling(n_lc_out, filt_sz, stride, cur_vc, name=name)
            self.lc_upsample.add_module(str(i), mod)
            cur_vc = mod.vc

        # This vc describes the bounds of the input wav corresponding to the
        # local conditioning vectors
        self.vc['last_upsample'] = cur_vc
        self.cond = Conditioning(n_speakers, n_global_embed)
        self.base_layer = Conv1dWrap('Base Layer', cur_vc, in_channels=n_quant,
                out_channels=n_res, kernel_size=1, stride=1, dilation=1,
                bias=self.bias)

        self.base_layer.vc.do_trim_input = True
        cur_vc = self.base_layer.vc

        self.conv_layers = nn.ModuleList() 
        n_cond = n_lc_out + n_global_embed

        for b in range(self.n_blocks):
            for bl in range(self.n_block_layers):
                dil = 2**bl
                name = 'GRCC_{},{}(dil={})'.format(b, bl, dil)
                grc = GatedResidualCondConv(self.vc, n_cond, n_res, n_dil,
                        n_skp, 1, dil, filter_sz, bias, cur_vc, name)
                self.conv_layers.append(grc)
                cur_vc = grc.vc

        # Each module in the stack needs to know the dimensions of
        # the input and output of the overall stack, in order to trim
        # residual connections
        self.vc['beg_grcc'] = self.conv_layers[0].vc
        self.vc['end_grcc'] = self.conv_layers[-1].vc 

        self.relu = nn.ReLU()
        self.post1 = Conv1dWrap('Post1', cur_vc, in_channels=n_skp,
                out_channels=n_post, kernel_size=1, stride=1, bias=bias)

        self.post2 = Conv1dWrap('Post2', self.post1.vc, in_channels=n_post,
                out_channels=n_quant, kernel_size=1, stride=1, bias=bias)
        self.logsoftmax = nn.LogSoftmax(1) # (B, Q, N)
        self.vc['main'] = self.post2.vc 

    def set_parent_vc(self, parent_vc):
        self.vc['beg'].parent = parent_vc
        parent_vc.child = self.vc['beg']

    def post_init(self):
        for grc in self.conv_layers:
            grc.post_init()

    def update_leads(self):
        for grc in self.conv_layers:
            grc.update_leads()

    def set_incremental(self):
        """
        Set cond_lead and skip_leads for incremental mode
        """
        for grc in self.conv_layers:
            grc.set_incremental()

    def forward(self, wav_onehot, lc_sparse, speaker_inds, jitter_index):
        """
        B: n_batch (# of separate wav streams being processed)
        T1: n_wav_timesteps
        T2: n_conditioning_timesteps
        I: n_in
        L: n_lc_in
        Q: n_quant

        wav: (B, Q, T1)
        lc: (B, L, T2)
        speaker_inds: (B, T)
        outputs: (B, Q, N)
        """
        D1 = lc_sparse.size()[1]
        lc_jitter = torch.take(lc_sparse,
                jitter_index.unsqueeze(1).expand(-1, D1, -1))
        lc_conv = self.lc_conv(lc_jitter) 
        lc_dense = self.lc_upsample(lc_conv)

        # Trimming due to different phases of the input MFCC windows
        # W2 = lcond_slice.size()[1] 

        D2 = lc_dense.size()[1]
        lc_dense_trim = lc_dense[:,:,self.trim_ups_out[0]:self.trim_ups_out[1]]
        # lc_dense_trim = torch.take(lc_dense,
        #         lcond_slice.unsqueeze(1).expand(-1, D2, -1))

        cond = self.cond(lc_dense_trim, speaker_inds)
        # "The conditioning signal was passed separately into each layer" - p 5 pp 1.
        # Oddly, they claim the global signal is just passed in as one-hot vectors.
        # But, this means wavenet's parameters would have N_s baked in, and wouldn't
        # be able to operate with a new speaker ID.

        sig = self.base_layer(wav_onehot) 
        sig, skp_sum = self.conv_layers[0](sig, cond)
        for layer in self.conv_layers[1:]:
            sig, skp = layer(sig, cond)
            skp_sum += skp
            
        post1 = self.post1(self.relu(skp_sum))
        quant = self.post2(self.relu(post1))
        # we only need this for inference time
        # logits = self.logsoftmax(quant) 
        return quant


    def sample(self, wav_onehot, lc_sparse, speaker_inds, jitter_index, n_rep):
        """
        Generate n_rep samples, using lc_sparse and speaker_inds for local and global
        conditioning.  

        wav_onehot: full length wav vector
        lc_sparse: full length local conditioning vector derived from full
        wav_onehot
        """
        # initialize model geometry
        mfcc_vc = self.vc['beg'].parent
        up_vc = self.vc['beg'].child
        beg_grcc_vc = self.vc['beg_grcc']
        end_vc = self.vc['end_grcc']

        # calculate full output range
        wav_gr = vconv.GridRange((0, int(1e12)), (0, wav_onehot.size()[2]), 1)
        full_out_gr = vconv.output_range(mfcc_vc, end_vc, wav_gr)
        n_ts = full_out_gr.sub_length()

        # calculate starting input range for single timestep
        one_gr = vconv.GridRange((0, int(1e12)), (0, 1), 1)
        vconv.compute_inputs(end_vc, one_gr)

        # calculate starting position of wav
        wav_beg = int(beg_grcc_vc.input_gr.sub[0] - mfcc_vc.input_gr.sub[0])
        # wav_end = int(beg_grcc_vc.input_gr.sub[1] - mfcc_vc.input_gr.sub[0])
        wav_onehot = wav_onehot[:,:,wav_beg:]

        # !!! hack - I'm not sure why the int() cast is necessary
        n_init_ts = int(beg_grcc_vc.in_len())

        lc_sparse = lc_sparse.repeat(n_rep, 1, 1)
        jitter_index = jitter_index.repeat(n_rep, 1)
        speaker_inds = speaker_inds.repeat(n_rep)

        # precalculate conditioning vector for all timesteps
        D1 = lc_sparse.size()[1]
        lc_jitter = torch.take(lc_sparse,
                jitter_index.unsqueeze(1).expand(-1, D1, -1))
        lc_conv = self.lc_conv(lc_jitter) 
        lc_dense = self.lc_upsample(lc_conv)
        cond = self.cond(lc_dense, speaker_inds)
        n_ts = cond.size()[2]

        chunk_size = 10000
        start_pos = 26000
        n_samples = 30000

        cur_pos = torch.tensor([start_pos], dtype=torch.int32,
                device=wav_onehot.device)
        end_pos = torch.tensor([start_pos + n_samples], dtype=torch.int32,
                device=wav_onehot.device)

        wav_onehot = wav_onehot.repeat(n_rep, 1, 1)
        n_layers = self.ac.n_blocks * self.ac.n_block_layers

        # sig[0] is the output of the base_layer
        # sig[i] is the output of the conv_layer[i-1]
        # there is no sig to hold the output of conv_layer[-1]
        # instead, it is directed to sig[n_layers-1]

        # wav_irng slices wav_onehot when used as input, and
        # we derive the single position output from wav_irng
        irng = wav_onehot.new_empty(n_layers, 2, dtype=torch.int32)

        # orng[l] is the output range of layer l, which populates sig[l]
        # except that orng[-2] and orng[-1] both populate sig[-1]
        # because
        orng = wav_onehot.new_empty(n_layers + 1, 2, dtype=torch.int32)

        # the one  
        cond_rng = wav_onehot.new_empty(2, dtype=torch.int32)

        # input range for the wav_onehot vector
        wav_ir = wav_onehot.new_empty(2, dtype=torch.int32)

        skp_sum = wav_onehot.new_zeros(n_rep, self.ac.n_skp, 1)

        # receptive field of a layer i relative to single output element
        # in final layer
        global_rf = []

        # receptive field size of layer i relative to single output element
        # in next layer
        local_rf = []

        vc = self.vc['beg_grcc'] 
        while vc != self.vc['end_grcc'].child:
            global_rf.append(vc.in_len())
            local_rf.append(vc.filter_size())
            vc = vc.child

        wav_ir[0] = cur_pos[0]
        wav_ir[1] = cur_pos[0] + global_rf[0]

        sig = [None] * n_layers
        for i in range(n_layers):
            sig[i] = wav_onehot.new_empty(n_rep, self.ac.n_res, global_rf[i] 
                + chunk_size - 1)
            irng[i,0] = 0 
            irng[i,1] = global_rf[i] 
            orng[i,0] = 0 
            orng[i,1] = global_rf[i]

        orng[-1,0] = 0
        orng[-1,1] = 1
        cond_rng[0] = 0
        cond_rng[1] = global_rf[0]
        
        while not torch.equal(end_pos, cur_pos):
            chunk_size = min(chunk_size, end_pos - cur_pos)

            for i in range(chunk_size):
                # base_layer is a 1x1 convolution, so uses irng[0] 
                # for both input and output
                ir = irng[0]
                sig[0][:,:,ir[0]:ir[1]] = \
                        self.base_layer(wav_onehot[:,:,wav_ir[0]:wav_ir[1]]) 

                for i, layer in enumerate(self.conv_layers):
                    # last iteration reassigns to same sig slot (unused) 
                    i_out = min(i+1, n_layers - 1)
                    p, q = irng[i], orng[i+1]
                    sig[i_out][:,:,q[0]:q[1]], skp = layer(
                            sig[i][:,:,p[0]:p[1]],
                            cond[:,:,cond_rng[0]:cond_rng[1]]
                        )
                    skp_sum += skp

                post1 = self.post1(self.relu(skp_sum))
                quant = self.post2(self.relu(post1))
                cat = dcat.OneHotCategorical(logits=quant.squeeze(2))
                wav_onehot[:,:,wav_ir[1]] = cat.sample()

                if irng[0,0] == 0:
                    # finished initialization, now incremental mode
                    # we only really need 1 new element, but computing two
                    # nicely fits with sig[0]
                    self.set_incremental()
                    cond_rng[0] = cond_rng[1] - 1
                    wav_ir[0] = wav_ir[1] - local_rf[0]

                    vc = self.vc['beg_grcc']
                    for i in range(n_layers):
                        irng[i,0] = global_rf[i] - local_rf[i] 
                        irng[i,1] = global_rf[i]
                        orng[i,0] = global_rf[i] - 1 
                        orng[i,1] = global_rf[i]
                        
                orng += 1
                irng += 1
                wav_ir += 1
                cond_rng += 1

                if irng[0,0] % 100 == 0:
                    print(irng[0], end_pos[0])

            # reset windows
            cur_pos += chunk_size
            for i in range(n_layers):
                sig[i][:,:,0:global_rf[i]] = \
                        sig[i][:,:,global_rf[i] + chunk_size]
                irng[i,0] = 0
                irng[i,1] = global_rf[i]
                orng[i,0] = 0
                orng[i,1] = global_rf[i]

            cond_rng -= chunk_size


        """
        while not torch.equal(inrange[1], end_ind[0]):
        # while inrange[1] != end_ind[0]:
            sig = self.base_layer(wav_onehot[:,:,inrange[0]:inrange[1]]) 
            sig, skp_sum = self.conv_layers[0](sig, cond[:,:,inrange[0]:inrange[1]])
            for layer in self.conv_layers[1:]:
                sig, skp = layer(sig, cond[:,:,inrange[0]:inrange[1]])
                skp_sum += skp

            post1 = self.post1(self.relu(skp_sum))
            quant = self.post2(self.relu(post1))
            cat = dcat.OneHotCategorical(logits=quant.squeeze(2))
            wav_onehot[1:,:,inrange[1]] = cat.sample()[1:,...]
            inrange += 1
            if inrange[0] % 100 == 0:
                print(inrange, end_ind[0])
        """

        
        # convert to value format
        quant_range = wav_onehot.new(list(range(self.n_quant)))
        wav = torch.matmul(wav_onehot.permute(0,2,1), quant_range)
        torch.set_printoptions(threshold=100000)
        pad = 5
        print('padding = {}'.format(pad))
        print('original')
        print(wav[0,start_pos-pad:end_pos+pad])
        print('synth')
        print(wav[1,start_pos-pad:end_pos+pad])

        # print(wav[:,end_pos:end_pos + 10000])
        print('synth range std: {}, baseline std: {}'.format(
            wav[:,start_pos:end_pos].std(), wav[:,end_pos:].std()
            ))

        return wav



class RecLoss(nn.Module):
    def __init__(self):
        super(RecLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(1) # input is (B, Q, N)

    def forward(self, quant_pred, target_wav):

        log_pred = self.logsoftmax(quant_pred)
        target_wav_gather = target_wav.long().unsqueeze(1)
        log_pred_target = torch.gather(log_pred, 1, target_wav_gather)

        rec_loss = - log_pred_target.mean()
        self.metrics = {
                'rec': rec_loss
                }

        return rec_loss 

