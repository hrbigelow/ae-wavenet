import fractions
import math
import numpy as np

"""
Some terminology
 Filter structure is conceptualized as a 'left wing, key element, and right
 wing'.
 
 S-input: spaced input.  The input tensor with spacing elements applied (for
 inverse strides)
 SP-input: spaced, padded input.  The input tensor with spacing elements and
 left/right flanking padding elements applied
 key-covered element:  For a given logical filter position over the input,
 this is the element that is covered by the 'key' filter element.
"""

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

        # Ensures that the key filter element is always over a non-padding
        # region.
        if (self.l_pad > self.l_wing_sz or self.r_pad > self.r_wing_sz):
            print(self)
            raise RuntimeError('Filter wing sizes cannot be less than the respective '
                    'padding')
        print(self)
        

    def __repr__(self):
        fmt = '[{}^{}, {}/{}, {}--{}, "{}"]'
        return fmt.format(
                self.l_wing_sz, self.r_wing_sz,
                self.stride_ratio.numerator, self.stride_ratio.denominator,
                self.l_pad, self.r_pad, self.name)

    def _get_rfield_lb(self, out_i):
        """
        Get the start of the input range which is the receptive field for the
        output element out_i.  
        """
        if self.stride_ratio >= 1:
            stride = self.stride_ratio.numerator
            # index in densified output
            out_dense_i = out_i * stride
            # index in P-input of key-covered element
            kc_in_p_i = out_dense_i + self.l_wing_sz
            # index in P-input of left-covered element (>= 0)
            lb_in_p_i = kc_in_p_i - self.l_wing_sz
            # index in input of left-covered element
            lb_in_i = lb_in_p_i - self.l_pad
            # final
            in_i = min(0, lb_in_i)
        else:
            inv_stride = self.stride_ratio.denominator
            # index in SP-input of key-covered element
            kc_i = out_i + self.l_wing_sz
            # index in SP-input of left-covered element
            lb_filt_i = kc_i - self.l_wing_sz
            # index in S-input of left-covered element
            unpad_i = max(0, lb_filt_i - self.l_pad)
            # index in input of left-covered element
            in_i = math.ceil(unpad_i / inv_stride)
        return in_i

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
            # index in densified output
            out_dense_i = out_i * stride
            # index in P-input of key-covered element
            kc_in_p_i = out_dense_i + self.l_wing_sz
            # index in P-input of right-covered element
            ub_in_p_i = kc_in_p_i + self.r_wing_sz
            # index in input of right-covered element
            ub_in_i = ub_in_p_i - self.l_pad
            # length calculation
            out_len = out_e + 1
            in_len = (out_len - 1) * stride + 1 - p + w
            in_i = min(ub_in_i, in_len - 1)
        else:
            inv_stride = self.stride_ratio.denominator
            # index in SP-input of key-covered element 
            kc_sp_in_i = out_i + self.l_wing_sz
            # index in SP-input of right-covered element
            ub_sp_in_i = kc_sp_in_i + self.r_wing_sz
            # index in S-input of right-covered element (>= 0)
            ub_s_in_i = ub_sp_in_i - self.l_pad
            # index in input of value element nearest right-covered element 
            ub_in_i = ub_s_in_i // inv_stride
            # length calculation
            out_len = out_e + 1
            in_len = (out_len + w - p - 1) // inv_stride + 1
            # final calculation
            in_i = min(ub_in_i, in_len - 1)
        return in_i


    def _get_ifield_lb(self, in_i):
        """
        Get the start of the output range which is the "influence field" for
        the input element in_i.  "influence field" is the inverse of the
        receptive field
        """
        w = self.l_wing_sz + self.r_wing_sz

        if self.stride_ratio >= 1:
            stride = self.stride_ratio.numerator
            # index in P-input
            p_in_i = in_i + self.l_pad
            # lower-bound ifield in P-input
            lb_p_in_i = max(0, p_in_i - self.r_wing_sz)
            # lower-bound ifield in dense output
            lb_dense_out_i = max(0, lb_p_in_i - self.l_wing_sz)
            # lower-bound ifield in strided output
            lb_out_i = math.ceil(lb_dense_out_i / stride)
            # final
            out_i = lb_out_i
        else:
            inv_stride = self.stride_ratio.denominator
            # index in S-input
            s_in_i = in_i * inv_stride
            # index in SP-input
            sp_in_i = s_in_i + self.l_pad
            # index in output using key-covered filter position
            kc_out_i = sp_in_i - self.l_wing_sz
            # index of output using right-covered filter position
            lb_out_i = max(0, kc_out_i - self.r_wing_sz)
            # final
            out_i = lb_out_i
        return out_i

    def _get_ifield_ub(self, in_i, in_e):
        """
        Get the end of the output range which is the "influence field" for the
        input element in_i.  in_e is the last element of the input. The
        "influence field" is the inverse of the receptive field.  We adopt the
        convention that any outputs at the end that arise solely from padding
        and spacing belong to the influence field of the last input element.
        """
        p = self.l_pad + self.r_pad
        w = self.l_wing_sz + self.r_wing_sz

        if self.stride_ratio >= 1:
            stride = self.stride_ratio.numerator
            # index in P-input
            p_in_i = in_i + self.l_pad 
            # index in P-input of output for left-covered filter position 
            ub_p_in_i = p_in_i + self.l_wing_sz
            # index in dense output for left-covered filter position 
            ub_dense_out_i = ub_p_in_i - self.l_wing_sz
            # index in strided output of left-covered element
            ub_out_i = ub_dense_out_i // stride
            # length calculation
            in_len = in_e + 1
            out_len = (in_len + p - w) // stride 
            # truncate the output if it is beyond the maximal length
            out_i = min(ub_out_i, out_len - 1)
        else:
            inv_stride = self.stride_ratio.denominator
            # index in S-input
            s_in_i = in_i * inv_stride
            # index in SP-input
            sp_in_i = s_in_i + self.l_pad
            # index in output using key-covered filter position 
            kc_out_i = sp_in_i - self.l_wing_sz
            # index in output using left-covered filter position
            ub_out_i = kc_out_i + self.l_wing_sz
            # length calculation
            in_len = in_e + 1
            out_len = (in_len - 1) * inv_stride + 1 + p - w
            # final calculation
            out_i = min(ub_out_i, out_len - 1)
        return out_i

def rfield(source, dest, out_b, out_e, out_len):
    """
    Calculate the input tensor index range receptive field of the output tensor
    range [out_b, out_e].  out_len is the length of the actual output.
    """
    # We need this check because there is no other convenient way to recognize
    # the empty interval.
    if out_b == out_e:
        return 0, 0

    vc = dest
    b, e, li = out_b, out_e, out_len - 1
    while True:
        b = vc._get_rfield_lb(b)
        e = vc._get_rfield_ub(e, li)
        li = vc._get_rfield_ub(li, li)
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
        e = vc._get_ifield_ub(e, l)
        l = vc._get_ifield_ub(l, l)
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

