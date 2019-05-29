import fractions
import math
import numpy as np

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
        self.child = None
        self.l_pad = padding[0]
        self.r_pad = padding[1]
        self.name = name
        self.src = None
        self.dst = None

        if self.parent is not None:
            self.parent.child = self

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

    def _get_rfield_start(self, out_i):
        """Get the start of the input range which is the receptive field for
        the output element out_i
        """
        if self.stride_ratio >= 1:
            stride = self.stride_ratio.numerator
            pad_i = out_i * stride
            in_i = max(0, pad_i - self.l_pad)
        else:
            stride = self.stride_ratio.denominator
            pad_i = out_i // stride
            in_i = max(0, pad_i - self.l_pad)
        return in_i

    def _get_rfield_end(self, out_i, out_e):
        """
        Get the end elements of the input receptive field for the given output
        elements out_i and out_e in tandem.  out_e is assumed to be the last
        element in the output, and is needed for correct calculation.
        """
        w = self.l_wing_sz + self.r_wing_sz
        p = self.l_pad + self.r_pad
        if self.stride_ratio >= 1:
            stride = self.stride_ratio.numerator
            pad_i = out_i * stride + w
            pad_e = out_e * stride + w
            in_e = pad_e - p
            in_i = min(max(0, pad_i - self.l_pad), in_e)
        else:
            inv_stride = self.stride_ratio.denominator
            pad_i = (out_i + w) // inv_stride
            pad_e = math.ceil((out_e + w) / inv_stride)
            in_e = pad_e - p
            in_i = min(max(0, pad_i - self.l_pad), in_e)
        return in_i, in_e

    def _ifield_start_aux(self, in_i, is_end):
        w = self.l_wing_sz + self.r_wing_sz
        if self.stride_ratio >= 1:
            stride = self.stride_ratio.numerator
            add = self.r_pad if is_end else 0
            out_i = (in_i + self.l_pad + add - w) // stride
        else:
            inv_stride = self.stride_ratio.denominator
            out_i = (in_i + self.l_pad) * inv_stride - w
        return out_i

    def _get_ifield_start(self, in_i, in_e):
        """Get the start of the output range which is the "influence field" for
        the input element in_i.  is_end indicates whether in_i is the last
        element of the input. The "influence field" is the inverse of the
        receptive field"""
        return (self._ifield_start_aux(in_i, in_i == in_e),
                self._ifield_start_aux(in_e, True))

    def _ifield_end_aux(self, in_i, is_end):
        add = self.r_pad if is_end else 0
        if self.stride_ratio >= 1:
            stride = self.stride_ratio.numerator
            out_i = (in_i + self.l_pad + add) * stride
        else:
            inv_stride = self.stride_ratio.denominator
            out_i = (in_i + self.l_pad + add) * inv_stride
        return out_i

    def _get_ifield_end(self, in_i, in_e):
        """Get the end of the output range which is the "influence field" for
        the input element in_i.  is_end indicates whether in_i is the last
        element of the input.  The "influence field" is the inverse of the
        receptive field"""
        return (self._ifield_end_aux(in_i, in_i == in_e),
                self._ifield_end_aux(in_e, True))

            
def rfield(source, dest, out_b, out_e, last_out):
    """
    Calculate the input tensor index range receptive field of the output tensor
    range [out_b, out_e].  last_out is the last index in the actual output.
    """
    rf = dest
    b, e, l = out_b, out_e, last_out
    while True:
        b = rf.get_rfield_start(b)
        e, l = rf.get_rfield_end(e, l)
        if rf is source:
            break
        rf = rf.parent
    return b, e

def ifield(source, dest, in_b, in_e, in_last):
    """
    Calculates the output tensor index range which is the field of influence
    for the input range [in_b, in_e].  in_last is the last index of the actual
    input.
    """
    rf = source
    b, e = in_b, in_e
    b_is_end = (b == in_last)
    e_is_end = (e == in_last)
    while True:
        b = rf.get_ifield_start(b, b_is_end)
        e = rf.get_ifield_end(e, e_is_end)
        if rf is dest:
            break
        rf = rf.child
    return b, e

def _rrange(source, dest, in_i, in_e, lcm_de):
    rf = source
    sp = lcm_de
    q, e = in_i, in_e
    pe = e * sp
    while True:
        pe = pe + e * sp
        assert (sp * rf.stride_ratio).denominator == 1
        sp = int(sp * rf.stride_ratio)
        pq = pe - (e - q) * sp
        q, e = rf.get_ifield_end(q, e)
        if rf is dest:
            break
        rf = rf.child
    reduce_sp = np.lcm.reduce(lcm_de, sp)
    assert pq % reduce_sp == 0
    assert pe % reduce_sp == 0
    return pq // reduce_sp

def rrange(source, dest, in_b, in_e, in_last):
    """
    Calculate the index range of the output corresponding to
    the physical position range of the input.
    """
    # Get tightest spacing
    de = []
    rf = source
    while True:
        de.append(rf.stride_ratio.denominator)
        if rf is dest:
            break
        rf = rf.child
    lcm_de = fractions.Fraction(np.lcm.reduce(de))
    out_b = _rrange(source, dest, in_b, in_last)
    out_e = _rrange(source, dest, in_e, in_last)
    return out_b, out_e

