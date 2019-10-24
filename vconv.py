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


    def _recep_field(self, out_b, out_e, out_l):
        """
        Virtually back-computes the convolution that results in output range
        [out_b, out_l].

        Returns (in_b, in_e, in_l) where:

        in_b: index of the left-most element in the input which is in the
        receptive field of [out_b, out_e].  If no element exists, returns -1

        in_e: index of the right-most element in the input which is in the
        receptive field of [out_b, out_e].  If no element exists, returns -1

        in_l: index of the right-most element in the input which is in the
        receptive field of [out_b, out_l].  If no element exists, returns -1
        """
        lw, rw = self.l_wing_sz, self.r_wing_sz
        w = lw + rw
        p = self.l_pad + self.r_pad
        assert p <= w # this is upheld by the constructor logic

        if self.stride_ratio >= 1:
            stride = self.stride_ratio.numerator

            # index in densified output
            out_dense_b = out_b * stride
            out_dense_e = out_e * stride
            out_dense_l = out_l * stride

            # index in P-input of RF lower bound 
            lb_in_p_b = out_dense_b + lw - lw
            # index in P-input of RF upper bounds 
            ub_in_p_e = out_dense_e + rw - lw
            ub_in_p_l = out_dense_l + rw - lw

            # index in input of RF lower bound
            lb_in_b = lb_in_p_b - self.l_pad
            ub_in_e = ub_in_p_e - self.l_pad
            ub_in_l = ub_in_p_l - self.l_pad

            # length calculation
            out_len = out_l + 1
            in_len = (out_len - 1) * stride + 1 - p + w

            # final
            in_b = max(0, lb_in_i)
            in_e = min(ub_in_e, in_len - 1)
            in_l = min(ub_in_l, in_len - 1)
        else
            inv_stride = self.stride_ratio.denominator
            # index in SP-input of bound 
            lb_sp_b = out_b + lw
            ub_sp_e = out_e + lw + rw
            ub_sp_l = out_l + lw + rw

            # index in S-input of bound
            lb_s_b = lb_sp_b - self.l_pad
            ub_s_e = ub_sp_e - self.l_pad
            ub_s_l = ub_sp_l - self.l_pad

            # length calculation for bounds correction
            out_len = out_l + 1
            # assumes no pad-adjacent spacing in the input 
            in_len = math.ceil((out_len + w - p - 1) / inv_stride) + 1

            # index in input of bound
            lb_b = math.ceil(lb_s_b / inv_stride)
            ub_e = ub_s_e // inv_stride
            ub_l = ub_s_l // inv_stride

            # final
            in_b = max(0, lb_b)
            in_e = min(ub_e, in_len - 1)
            in_l = min(ub_l, in_len - 1)

        return in_b, in_e, in_l



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
            in_i = max(0, lb_in_i)
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
            #      #############
            #   @@@#**#**#**#**#@@@@
            
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
            # assumes no pad-adjacent spacing in the input 
            in_len = math.ceil((out_len + w - p - 1) / inv_stride) + 1
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
            out_len = (in_len + p - w - 1) // stride + 1
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

    def _output_range(self, in_b, in_e, in_l):
        """
        Virtually computes the convolution result given input range [0, in_l].
        
        Returns (out_b, out_e, out_l) where:

        out_b: index of the left-most element containing input[in_b] in its
        receptive field, and with the entire receptive field a subset of [in_b,
        in_e], or -1 if no such element exists.
        
        out_e: index of the right-most element containing input[in_e] in its
        receptive field, and with the entire receptive field a subset of [in_b,
        in_e] or -1 if no such element exists.

        out_l: inex of the right-most element containing input[in_l] in its
        receptive field, and with the entire receptive field a subset of
        [in_b, in_l] or -1 if no such element exists.
        """
        lw, rw = self.l_wing_sz, self.r_wing_sz
        w = lw + rw
        p = self.l_pad + self.r_pad

        if self.stride_ratio >= 1:
            stride = self.stride_ratio.numerator
            # right-most index in P-input that contains input[in_e] in its
            # receptive field
            p_in_rt_i = self.l_pad + in_e + int(in_e == in_l) * self.r_pad
            p_in_rt_last_i = self.l_pad + in_l + self.r_pad
            # index in dense output of upper-bound 
            ub_dense_i = p_in_rt_i - rw - lw
            ub_dense_last_i = p_in_rt_last_i - rw - lw
            # final.  -1 signals that no output exists satisfhying criteria
            ub_i = max(-1, ub_dense_i // stride)
            ub_last_i = max(-1, ub_dense_last_i // stride)
            # left-most index in P-input that contains input[in_b] in its
            # receptive field
            p_in_lf_i = in_b - int(in_b == 0) * self.l_pad
            # filter must fit
            if p_in_lf_i + lw + rw > in_e:
                lb_i = -1
            lb_dense_i = p_in_lf_i + lw - lw
            lb_i = math.ceil(lb_dense_i / stride)
        else:
            inv_stride = self.stride_ratio.denominator
            s_in_rt_i = in_e * inv_stride
            s_in_rt_last_i = in_l * inv_stride
            # right-most index in SP-input
            # can use right-padding for the last element
            sp_in_rt_i = (s_in_rt_i + self.l_pad + int(in_e == in_l) *
                    self.r_pad)
            sp_in_rt_last_i = (s_in_rt_last_i + self.l_pad + self.r_pad)
            # final
            ub_i = max(-1, sp_in_rt_i - lw - rw)
            ub_last_i = max(-1, sp_in_rt_last_i - lw - rw)
            s_in_lf_i = in_b * inv_stride
            # index in SP-input of left-most element usable for input[in_b]
            sp_in_lf_i = s_in_lf_i + int(in_b != 0) * self.l_pad
            # length
            sp_in_len = in_l * inv_stride + p + 1
            if sp_in_lf_i + lw + rw >= sp_in_len:
                lb_i = -1
            else:
                lb_i = sp_in_lf_i + lw - lw

        return lb_i, ub_i, ub_last_i



def recep_field(source, dest, out_b, out_e, out_len):
    """
    Calculate the input tensor index range [in_b, in_e) receptive field of the
    output tensor range [out_b, out_e). out_len is the length of the actual
    output.
    Returns in_b, in_e, input_length
    """
    # We need this check because there is no other convenient way to recognize
    # the empty interval.
    if out_b == out_e:
        return 0, 0, 0

    vc = dest
    b, e, l = out_b, out_e - 1, out_len - 1
    while True:
        b_p, e_p = b, e
        b, e, l = vc._recep_field(b, e, l)
        print('in: [{}, {}), out: [{}, {}), {}'.format(b_p, e_p + 1, b, e + 1, vc))
        if vc is source:
            break
        vc = vc.parent
    return b, e + 1, l + 1


def output_range(source, dest, in_b, in_e, in_len):
    """
    Calculates the maximal set of output elements [out_b, out_e) in which
    each element has a receptive field which is a subset of [in_b, in_e)
    if in_b == 0 or in_e == in_len, the operation is allowed to make use of
    left or right padding, respectively.
    Will return [0, 0) if there are no output elements that satisfy the
    criteria.
    """
    if in_b == in_e:
        return 0, 0, 0

    vc = source
    b, e, l = in_b, in_e - 1, in_len - 1
    while True:
        b_p, e_p = b, e
        b, e, l = vc._output_range(b, e, l)
        print('in: [{}, {}), out: [{}, {}), {}'.format(b_p, e_p + 1, b, e + 1, vc))
        if vc is dest:
            break
        vc = vc.child
    return b, e + 1, l + 1



def ifield(source, dest, in_b, in_e, in_len):
    """
    Calculates the output tensor index range [out_b, out_e) which is the field of
    influence for the input range [in_b, in_e).
    """
    # We need this check because there is no other convenient way to recognize
    # the empty interval.
    if in_b == in_e:
        return 0, 0

    vc = source
    b, e, l = in_b, in_e - 1, in_len - 1
    while True:
        b_p, e_p = b, e
        b = vc._get_ifield_lb(b) 
        e = vc._get_ifield_ub(e, l)
        l = vc._get_ifield_ub(l, l)
        print('in: [{}, {}), out: [{}, {}), {}'.format(b_p, e_p + 1, b, e + 1, vc))
        if vc is dest:
            break
        vc = vc.child
    return b, e + 1

def _shadow(source, dest, in_b, in_e, in_len, spacing_denom_lcm):
    """
    Finds the shadow range in the input corresponding to the input range [in_b,
    in_e).
    """
    vc = source
    sp = spacing_denom_lcm
    b, e, l = in_b, in_e - 1, in_len - 1 
    # position of first element of shadow range 
    pf = 0
    while True:
        b_prev = b
        e_prev = e
        b, e, l = vc._output_range(b, e, l)
        # Current spacing should always be integral
        assert (sp * vc.stride_ratio).denominator == 1
        sp_prev = sp
        sp = int(sp * vc.stride_ratio)
        pf_prev = pf
        pf = pf + (vc.l_wing_sz - vc.l_pad) * sp_prev
        print('sp: ({}, {}), b: ({}, {}), p: ({}, {}), pf: ({}, {}), {}'.format(
            sp_prev, sp, b_prev, b, e_prev, e, pf_prev, pf, vc))
        if vc is dest:
            break
        vc = vc.child
    pb = pf + b * sp
    pe = pf + e * sp
    reduce_sp = np.gcd(spacing_denom_lcm, sp)
    assert pb % reduce_sp == 0
    assert pe % reduce_sp == 0
    return pb // reduce_sp , pe // reduce_sp + 1

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
    spacing = fractions.Fraction(1, 1)
    while True:
        spacing *= vc.stride_ratio
        de.append(spacing.denominator)
        if vc is dest:
            break
        vc = vc.child
    # the least common multiple of the running product of spacings
    spacing_denom_lcm = np.lcm.reduce(de)
    shadow_b, shadow_e = _shadow(source, dest, in_b, in_e, in_l,
            spacing_denom_lcm)
    return shadow_b, shadow_e 

