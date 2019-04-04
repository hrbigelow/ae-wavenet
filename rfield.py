# An instance of this class represents the coordinate relationship between an
# output element and its input receptive field.
import fractions 
import numpy as np
import math

class _Stats(object):
    '''Describes a 1D tensor of positioned elements''' 
    def __init__(self, l_pad, r_pad, n_val_elem, spc, vspc, l_pos, r_pos,
            src=None, dst=None):
        self.l_pad = l_pad
        self.r_pad = r_pad
        self.nv = n_val_elem 
        self.spc = spc
        self.vspc = vspc
        self.l_pos = l_pos
        self.r_pos = r_pos
        self.src = src
        self.dst = dst

    def __repr__(self):
        src_name = 'None' if self.src is None else self.src.name
        dst_name = 'None' if self.dst is None else self.dst.name

        return 'l_pad: {}, r_pad: {}, nv: {}, spc: {}, vspc: {}, l_pos: {}, ' \
                'r_pos: {}, span(): {}, src: {}, dst: {}\n'.format(
                        self.l_pad, self.r_pad, self.nv, self.spc, self.vspc,
                        self.l_pos, self.r_pos, self.span(), src_name, dst_name)

    def symbolic(self):
        in_pad = (self.vspc // self.spc) - 1
        spc_str = ' {}P '.format(in_pad)
        body = spc_str.join(['V'] * self.nv) 
        l_pad = '{}L'.format(self.l_pad)
        r_pad = '{}R'.format(self.r_pad)
        return ' '.join([l_pad, body, r_pad])

    def span(self):
        return (self.nv - 1) * self.vspc + self.l_pos - self.r_pos + 1

def _normalize_stats_(stats):
    '''Adjusts all spacing to be non-fractional, and aligns left and right
    positions'''
    lcm_denom = fractions.Fraction(np.lcm.reduce(tuple(s.spc.denominator for s
        in stats)))

    for s in stats:
        s.spc = int(s.spc * lcm_denom)
        s.vspc = int(s.vspc * lcm_denom)
        s.l_pos = int(s.l_pos * lcm_denom)
        s.r_pos = int(s.r_pos * lcm_denom)

    min_l_pos = min(s.l_pos for s in stats)
    max_r_pos = max(s.r_pos for s in stats)

    for s in stats: 
        s.l_pos = s.l_pos - min_l_pos
        s.r_pos = s.r_pos - max_r_pos

def print_stats(stats, xpad='x', ipad='o', data='*'):
    '''pretty print a symbolic stack of 1D window-based tensor calculations
    represented by 'stats', showing strides, padding, and dilation.  Generate
    stats using Rfield::get_stats()'''

    def _dilate_string(string, space_sym, spacing):
        if len(string) == 0:
            return ''
        space_str = space_sym * (spacing - 1)
        return string[0] + ''.join(space_str + s for s in string[1:])

    def l_pad_pos(st):
        return st.l_pos - st.l_pad * st.spc

    def r_pad_pos(st):
        return st.r_pos + st.r_pad * st.spc

    min_pad_pos = min(l_pad_pos(st) for st in stats)
    max_pad_pos = max(r_pad_pos(st) for st in stats)

    print('\n')
    for st in stats:
        l_spc = ' ' * (l_pad_pos(st) - min_pad_pos)
        r_spc = ' ' * (max_pad_pos - r_pad_pos(st))
        core = _dilate_string(data * st.nv, ipad, round(st.vspc / st.spc))
        body = xpad * st.l_pad + core + xpad * st.r_pad
        body_dil = _dilate_string(body, ' ', round(st.spc))
        s = '|' + l_spc + body_dil + r_spc + '|'
        print(s)


class Rfield(object):
    '''
    An instance of Rfield describes one 1D scanning-window transformation
    such as a convolution, transpose convolution or fourier transform.  Chain
    instances together with the 'parent' field, to represent a feed-forward
    chain of transformations.

    Then, use the final child to calculate needed input size for a desired
    output size, and offsets relative to the input.
    '''
    def __init__(self, filter_info, padding=(0, 0), stride=1,
            is_downsample=True, parent=None, name=None):
        self.parent = parent
        self.l_pad = padding[0]
        self.r_pad = padding[1]
        self.name = name

        # stride_ratio is ratio of output spacing to input spacing
        if is_downsample:
            self.stride_ratio = fractions.Fraction(stride, 1)
        else:
            self.stride_ratio = fractions.Fraction(1, stride)

        if isinstance(filter_info, tuple):
            self.l_wing_sz = filter_info[0]
            self.r_wing_sz = filter_info[1]
        elif isinstance(filter_info, int):
            total_wing_sz = filter_info - 1
            self.l_wing_sz = total_wing_sz // 2
            self.r_wing_sz = total_wing_sz - self.l_wing_sz
        else:
            raise RuntimeError('filter_info must be either a 2-tuple of '
                    '(l_wing_sz, r_wing_sz) or an integer of filter_sz')

    def __repr__(self):
        return 'name: {}, wing_sizes: {}, stride_ratio: {}, padding: {}\n'.format(
                self.name, (self.l_wing_sz, self.r_wing_sz),
                self.stride_ratio, (self.l_pad, self.r_pad))

    def _in_stride(self, output_stride_r):
        return output_stride_r / self.stride_ratio

    def _in_spacing(self, out_spc, out_vspc):
        if self.stride_ratio > 1:
            in_spc = out_vspc / self.stride_ratio
            in_vspc = in_spc
        else:
            in_spc = out_vspc
            in_vspc = in_spc / self.stride_ratio
        return in_spc, in_vspc

    def _local_bounds(self):
        '''For this transformation, calculates the offset between the first
        value elements of the input and the output, assuming output element
        spacing of 1.  Return values must be adjusted by the actual output
        element spacing.
        '''
        # spacing is the distance between any two consecutive elements in this
        # tensor, including padding elements 
        l_off = self.l_wing_sz - self.l_pad
        r_off = self.r_wing_sz - self.r_pad
        return l_off, r_off

    def chain_length(self, other=None):
        '''Get number of links between self and other, or raise an error.
        set other=None to get full chain length'''
        n, cur = 0, self
        while cur != other and cur != None:
            n += 1
            cur = cur.parent
        if cur != other:
            raise RuntimeError('chain ended and stop node not found')
        return n

    def _get_chain(self, chain=None):
        if chain is None:
            chain = []
        cur = self
        while cur is not None:
            chain.append(cur)
            cur = cur.parent
        return chain

    def get_chain(self):
        '''Get all transformations in child->parent order'''
        return self._get_chain()

    def _resolve_stop(self, stop_at):
        if isinstance(stop_at, Rfield) or stop_at is None:
            try:
                depth = self.chain_length(stop_at)
            except RuntimeError:
                raise RuntimeError('Did not find "stop_at" node in the chain of parents')
        elif isinstance(stop_at, int):
            depth = stop_at
        else:
            raise RuntimeError('"stop_at" must be int (number of levels) or Rfield '
                    '(node to stop at)')
        return depth

    def _num_in_elem(self, n_out_elem_req):
        '''calculate the number of input value elements (not counting padding
        or dilation elements) needed to produce at least the desired number of
        output elements.  Note that the number of input elements needed to
        produce at least n_out_elem_req may in fact produce more than that.'''
        # See rfield_notes.txt for formulas
        def _spaced(n_el, stride):
            return (n_el - 1) * stride + 1

        lw, rw = self.l_wing_sz, self.r_wing_sz
        lp, rp = self.l_pad, self.r_pad

        # Or: output requested, I: input size needed
        # downsampling: LW + spaced(Or, S) + RW = LP + spaced(I, 1) + RP
        # upsampling  : LW + spaced(Or, 1) + RW = LP + spaced(I, S) + RP
        const = lw + rw - lp - rp

        if self.stride_ratio.denominator == 1:
            # downsampling
            stride = self.stride_ratio.numerator
            n_in_elem = (n_out_elem_req - 1) * stride + 1 + const
        else:
            # upsampling
            stride = self.stride_ratio.denominator
            n_in_elem = math.ceil((n_out_elem_req + stride - 1 + const) / stride)
        return n_in_elem

    def _num_out_elem(self, n_in_elem):
        '''Calculate number of output elements for the given number of
        input elements.'''
        def _spaced(n_el, stride):
            return (n_el - 1) * stride + 1

        lw, rw = self.l_wing_sz, self.r_wing_sz
        lp, rp = self.l_pad, self.r_pad
        const = lw + rw - lp - rp

        if self.stride_ratio.denominator == 1:
            # downsampling
            stride = self.stride_ratio.numerator
            n_out_elem = (n_in_elem - const + stride - 1) // stride
        else:
            # upsampling
            stride = self.stride_ratio.denominator
            n_out_elem = _spaced(n_in_elem, stride) - const
        return n_out_elem

    def get_stats(self, n_out_el, stop_at=None, out_spc=1, out_vspc=1, l_out_pos=0,
            r_out_pos=0, accu=None, is_valid=True):
        '''Return a tuple: (stats, is_valid).  stats is an array of items which
        describe each tensor produced in the chain of transformations.
        is_valid will be false if at any point in the chain, the number of
        value elements is <= 0.  '''
        depth = self._resolve_stop(stop_at)

        if accu is None:
            first = _Stats(0, 0, n_out_el, out_spc, out_vspc, l_out_pos, r_out_pos,
                    src=self, dst=None)
            accu = [first]

        l_off, r_off = self._local_bounds()
        n_in_el = self._num_in_elem(n_out_el)
        is_valid &= n_out_el > 0
        in_spc, in_vspc = self._in_spacing(out_spc, out_vspc)
        l_in_pos = l_out_pos - l_off * in_spc
        r_in_pos = r_out_pos + r_off * in_spc

        stats = _Stats(self.l_pad, self.r_pad, n_in_el, in_spc, in_vspc,
                l_in_pos, r_in_pos, src=self.parent, dst=self)
        accu.append(stats)

        if depth == 1:
            # Expand number of value elements
            for i in reversed(range(len(accu) - 1)):
                cur, pre = accu[i], accu[i+1]
                cur.nv = pre.dst._num_out_elem(pre.nv)
            _normalize_stats_(accu)
            return accu, is_valid
        else:
            return self.parent.get_stats(n_in_el, depth - 1, in_spc, in_vspc,
                    l_in_pos, r_in_pos, accu, is_valid)

    def geometry(self, n_out_elem_req, stop_at=None):
        '''For this stack of transformations, calculate number of input
        elements needed to generate at least n_out_elem_req (number of
        requested output elements).  Also calculates begin index and end index,
        which are the elements in the input that correspond to the first and
        last elements in the output.

        stop_at is: 
        integer: the number of transformations to process
        RField: stop here without processing.
        None: process all transformations

        To process 1 transformation, pass: stop_at=1 or stop_at=self.parent

        To chain together different models, i.e. modelA -> modelB -> modelC
        modelb_n_out_elem, _ = modelC.rf.geometry(modelc_n_out_elem, stop_at=modelB.rf)
        modela_n_out_elem, _ = modelB.rf.geometry(modelb_n_out_elem, stop_at=modelA.rf)

        This is useful for coordinating inputs for models. 
        Returns num_input_el, num_output_el_actual, left_offset, right_offset, is_valid.

        num_input_el: number of input elements needed for the requested n_out_elem.
        left_offset: index in the input where the first element of the output aligns.
        right_offset: index in the input where the last element of the output aligns.
        is_valid: flag indicating whether the overall architecture of the model

        Any architecture in which a layer produces no output is considered
        invalid.  This will occur if there is too much padding specified, or
        n_out_elem_req is too low.
        '''
        stats, is_valid = self.get_stats(n_out_elem_req, stop_at=stop_at)
        inp = stats[-1]
        out = stats[0]
        return inp.nv, out.nv, out.l_pos - inp.l_pos, out.r_pos - inp.r_pos, is_valid

