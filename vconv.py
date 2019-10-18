import fractions
import math
import numpy as np

class VirtualConv(object):
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

        if (self.l_pad > self.l_wing_sz or
                self.r_pad > self.r_wing_sz):
            raise RuntimeError('Filter wing sizes cannot be less than the respective '
                    'padding')
        

        fmt = '[{}, {}, {}, {}, {}, {}, "{}"]'
        print(fmt.format(self.l_wing_sz, self.r_wing_sz,
            self.stride_ratio.numerator, self.stride_ratio.denominator,
            self.l_pad, self.r_pad,
            name))

    def __repr__(self):
        return 'name: {}, wing_sizes: {}, stride_ratio: {}, padding: {}\n'.format(
                self.name, (self.l_wing_sz, self.r_wing_sz),
                self.stride_ratio, (self.l_pad, self.r_pad))

    def _get_rfield_lb(self, out_i):
        """
        Get the start of the input range which is the receptive field for the
        output element out_i.  spc_i is the index in the spaced padded input
        that corresponds to out_i's left-most receptive field element.
        """
        if self.stride_ratio >= 1:
            stride = self.stride_ratio.numerator
            spc_i = out_i * stride
            in_i = max(0, spc_i - self.l_pad)
        else:
            stride = self.stride_ratio.denominator
            spc_i = out_i // stride
            in_i = max(0, spc_i - self.l_pad)
        return in_i

    # !!! fix this to agree with bounds logic
    def _get_rfield_ub(self, out_i, out_e):
        """
        Get the end element of the input receptive field for the given output
        elements out_i and out_e in tandem.  out_e is assumed to be the last
        element in the output, and is needed for correct calculation.
        """
        w = self.l_wing_sz + self.r_wing_sz
        p = self.l_pad + self.r_pad
        assert p <= w # this is upheld by the constructor logic

        if self.stride_ratio >= 1:
            stride = self.stride_ratio.numerator
            spc_i = out_i * stride + w
            spc_e = out_e * stride + w
            in_e = spc_e - p
            in_i = min(max(0, spc_i - self.l_pad), in_e)
        else:
            inv_stride = self.stride_ratio.denominator
            spc_i = (out_i + w) // inv_stride
            spc_e = math.ceil((out_e + w) / inv_stride)
            in_e = spc_e - p
            in_i = min(max(0, spc_i - self.l_pad), in_e)
        return in_i


    def _get_ifield_lb(self, in_i):
        """
        Get the start of the output range which is the "influence field" for
        the input element in_i.  "influence field" is the inverse of the
        receptive field
        """
        w = self.l_wing_sz + self.r_wing_sz
        if self.stride_ratio >= 1:
            sub = max(0, self.l_pad - w) if in_i == 0 else 0
            stride = self.stride_ratio.numerator
            out_i = math.ceil(max(0, in_i + self.l_pad - sub - w) / stride)
        else:
            sp_l_pad = (self.l_pad - 1) * inv_stride + 1
            sub = max(0, sp_l_pad - w) if in_i == 0 else 0 
            inv_stride = self.stride_ratio.denominator
            out_i = max(0, (in_i + self.l_pad) * inv_stride - sub - w)
        return out_i

    def _get_ifield_ub(self, in_i, in_l):
        """
        Get the end of the output range which is the "influence field" for the
        input element in_i.  in_l is input length The "influence field" is the
        inverse of the receptive field.  We adopt the convention that any
        outputs at the end that arise solely from padding and spacing belong to
        the influence field of the last input element.
        """
        p = self.l_pad + self.r_pad
        w = self.l_wing_sz + self.r_wing_sz
        if self.stride_ratio >= 1:
            add = max(0, self.r_pad - w) if in_i == in_l - 1 else 0
            stride = self.stride_ratio.numerator
            out_i = math.ceil(min(in_i + self.l_pad + add + 1, in_l + p - w) / stride)
        else:
            inv_stride = self.stride_ratio.denominator
            sp_r_pad = (self.r_pad - 1) * inv_stride + 1
            add = max(0, sp_r_pad - w) if in_i == (in_l - 1) else 0 
            out_i = (in_i + self.l_pad) * inv_stride + add + 1
        return out_i

def rfield(source, dest, out_b, out_e, out_l):
    """
    Calculate the input tensor index range receptive field of the output tensor
    range [out_b, out_e].  out_l is the last index in the actual output.
    """
    # We need this check because there is no other convenient way to recognize
    # the empty interval.
    if out_b == out_e:
        return 0, 0

    vc = dest
    b, e, l = out_b, out_e, out_l
    while True:
        b = vc._get_rfield_lb(b)
        e = vc._get_rfield_ub(e - 1, l)
        l = vc._get_rfield_ub(l - 1, l)
        if vc is source:
            break
        vc = vc.parent
    return b, e

def ifield(source, dest, in_b, in_e, in_l):
    """
    Calculates the output tensor index range which is the field of influence
    for the input range [in_b, in_e].  in_l is the length of the input.
    """
    # We need this check because there is no other convenient way to recognize
    # the empty interval.
    if in_b == in_e:
        return 0, 0

    vc = source
    b, e, l = in_b, in_e, in_l
    while True:
        b = vc._get_ifield_lb(b) 
        e = vc._get_ifield_ub(e - 1, l)
        l = vc._get_ifield_ub(l - 1, l)
        if vc is dest:
            break
        vc = vc.child
    return b, e

def _shadow(source, dest, in_b, in_e, in_l, lcm_de):
    vc = source
    sp = lcm_de
    b, e, l = in_b, in_e, in_l
    pf = 0
    while True:
        b = vc._get_ifield_lb(b, l)
        e = vc._get_ifield_ub(e, l)
        l = vc._get_ifield_ub(l, l)
        assert (sp * vc.stride_ratio).denominator == 1
        sp_prev = sp
        sp = int(sp_prev * vc.stride_ratio)
        pf = pf - (vc.l_pad * sp_prev) + (vc.l_wing_sz * sp)
        pb = pf + b * sp
        pe = pf + e * sp
        if vc is dest:
            break
        vc = vc.child
    reduce_sp = np.lcm.reduce(lcm_de, sp)
    assert pb % reduce_sp == 0
    assert pe % reduce_sp == 0
    return pb // reduce_sp, pe // reduce_sp

def shadow(source, dest, in_b, in_e, in_l):
    """
    Calculate the index range [shadow_in_b, shadow_in_e) of the input
    corresponding to the physical position range of the output produced by
    input range [in_b, in_e).  This could be thought of as the "shadow" of the
    output range on the input.
    """
    # Get tightest spacing
    de = []
    vc = source
    while True:
        de.append(vc.stride_ratio.denominator)
        if vc is dest:
            break
        vc = vc.child
    lcm_de = fractions.Fraction(np.lcm.reduce(de))
    shadow_b, shadow_e = _shadow(source, dest, in_b, in_e, in_l, lcm_de)
    return shadow_b, shadow_e 

