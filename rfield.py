# An instance of this class represents the coordinate relationship between an
# output element and its input receptive field.
import fractions 
import numpy as np

class _Stats(object):
    '''Describes a 1D tensor of positioned elements''' 
    def __init__(self, l_pad, r_pad, n_val_elem, spc, vspc, l_pos, r_pos):
        self.l_pad = l_pad
        self.r_pad = r_pad
        self.nv = n_val_elem 
        self.spc = spc
        self.vspc = vspc
        self.l_pos = l_pos
        self.r_pos = r_pos

    def __repr__(self):
        return 'l_pad: {}, r_pad: {}, n_value_elems: {}, spc: {}, vspc: {}, l_pos:' \
                ' {}, r_pos: {}'.format(self.l_pad, self.r_pad, self.nv, self.spc,
                        self.vspc, self.l_pos, self.r_pos)


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

def print_stats(stats, lpad='<', rpad='>', ipad='o', data='*'):
    '''pretty print a symbolic stack of 1D window-based tensor calculations
    represented by 'stats', showing strides, padding, and dilation.  Generate
    stats using FieldOffset::collect_stats()'''

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
        body = lpad * st.l_pad + core + rpad * st.r_pad
        body_dil = _dilate_string(body, ' ', round(st.spc))
        s = '|' + l_spc + body_dil + r_spc + '|'
        print(s)


class FieldOffset(object):
    '''
    Use this class as a member in individual Convolution or Transpose
    convolution modules.  Allows convenient back-calculation of needed input
    size for a desired output size.  Also, allows calculating the left and
    right offsets of the output from the input across the whole chain of
    transformations.

    '''
    def __init__(self, filter_info, padding=(0, 0), stride=1,
            is_downsample=True, parent=None):
        self.parent = parent
        self.l_pad = padding[0]
        self.r_pad = padding[1]

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
        return 'wing_sizes: {}, stride_ratio: {}, padding: {}'.format(
                (self.l_wing_sz, self.r_wing_sz), self.stride_ratio,
                (self.l_pad, self.r_pad))

    def print_chain(self):
        '''Recursively print the chain of transformations'''
        cur = self
        while cur is not None:
            print(cur)
            cur = cur.parent

    def num_levels(self):
        if self.parent is None:
            return 1
        else:
            return 1 + self.parent.num_levels()


    def _num_in_elem(self, n_out_elem):
        '''calculate the number of input value elements (not counting padding
        or dilation elements) needed to produce the desired number of output
        elements'''
        const = self.l_wing_sz + self.r_wing_sz - self.l_pad - self.r_pad
        if self.stride_ratio.denominator == 1:
            n_in_elem = (n_out_elem - 1) * self.stride_ratio.numerator + 1 + const
        else:
            n_in_elem = (self.stride_ratio.denominator + n_out_elem - 1 + const) \
                    // self.stride_ratio.denominator
        return n_in_elem        

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
        #input_spacing = max(fractions.Fraction(1, 1), self.stride_ratio)
        input_spacing = 1
        l_ind = self.l_wing_sz - self.l_pad
        r_ind = self.r_wing_sz - self.r_pad
        l_off = l_ind * input_spacing 
        r_off = r_ind * input_spacing 
        return l_off, r_off

    def collect_stats(self, n_out_el, out_spc=1, out_vspc=1, l_out_pos=0,
            r_out_pos=0, accu=None, is_valid=True):
        '''Return a tuple stats, is_valid.  stats is an array of items which
        describe each tensor produced in the chain of transformations.
        is_valid will be false if at any point in the chain, the number of
        value elements is <= 0.  '''
        l_off, r_off = self._local_bounds()
        n_in_el = self._num_in_elem(n_out_el)
        in_spc, in_vspc = self._in_spacing(out_spc, out_vspc)
        l_in_pos = l_out_pos - l_off * in_spc
        r_in_pos = r_out_pos + r_off * in_spc

        if accu is None:
            first = _Stats(0, 0, n_out_el, out_spc, out_vspc, l_out_pos,
                    r_out_pos)
            accu = [first]

        stats = _Stats(self.l_pad, self.r_pad, n_in_el, in_spc, in_vspc,
                l_in_pos, r_in_pos)
        accu.append(stats)

        if self.parent is None:
            _normalize_stats_(accu)
            return accu, is_valid 
        else:
            is_valid &= n_in_el > 0
            return self.parent.collect_stats(n_in_el, in_spc, in_vspc,
                    l_in_pos, r_in_pos, accu, is_valid)

    def geometry(self, n_out_elem):
        '''For this stack of transformations, calculate number of input
        elements needed to generate n_out_elem (number of output elements).
        Also calculates begin index and end index, which are the elements in
        the input that correspond to the first and last elements in the output.

        This is useful for coordinating inputs for models. 
        '''
        stats, is_valid = self.collect_stats(n_out_elem)
        inp = stats[-1]
        out = stats[0]
        return inp.nv, out.l_pos - inp.l_pos, out.r_pos - inp.r_pos, is_valid

