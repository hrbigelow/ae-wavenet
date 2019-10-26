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
        lp, rp = self.l_pad, self.r_pad
        w = lw + rw
        p = lp + rp
        assert p <= w # this is upheld by the constructor logic

        if self.stride_ratio >= 1:
            stride = self.stride_ratio.numerator

            # index in densified output
            out_dense_b = out_b * stride
            out_dense_e = out_e * stride
            out_dense_l = out_l * stride

            # index in P-input of RF bound 
            p_in_b = out_dense_b - lw + lw
            p_in_e = out_dense_e + rw + lw
            p_in_l = out_dense_l + rw + lw

            in_b = max(0, p_in_b - lp)
            in_l = p_in_l - lp - rp
            in_e = min(p_in_e - lp, in_l)

        else:
            inv_st = self.stride_ratio.denominator
            # index in SP-input of bound 
            sp_in_b = out_b - lw + lw
            sp_in_e = out_e + rw + lw
            sp_in_l = out_l + rw + lw

            # index in S-input of bound
            s_in_b = max(0, sp_in_b - lp)
            s_in_l = out_l - lw - rw
            s_in_e = min(sp_in_e - lp, s_in_l)

            # project to input
            in_b = math.ceil(s_in_b / inv_st)
            in_e = s_in_e // inv_st
            in_l = s_in_l // inv_st

        return in_b, in_e, in_l

    def _output_range(self, full_in, sub_in, gs_in):
        full_in_b, full_in_e = full_in
        sub_in_b, sub_in_e = sub_in
        gs_out = gs_in * self.stride_ratio
        assert gs_out == int(gs_out)
        gs_out = int(gs_out)
        empty = (-1, -1), (-1, -1), gs_out

        if self.stride_ratio >= 1:
            lpg = self.l_pad * gs_in
            rpg = self.r_pad * gs_in
            lwg = self.l_wing_sz * gs_in
            rwg = self.r_wing_sz * gs_in
            full_in_adj_b = full_in_b - lpg 
            full_in_adj_e = full_in_e + rpg
            if full_in_adj_e - full_in_adj_b < lwg + rwg:
                return empty
            if sub_in_e - sub_in_b < lwg + rwg:
                return empty
            full_out_b = full_in_adj_b + lwg
            full_out_pre_e = full_in_adj_e - rwg
            full_out_e = full_out_pre_e - (full_out_pre_e - full_out_b) % gs_out
            sub_out_pre_b = sub_in_b + lwg
            sub_out_pre_e = sub_in_e - rwg
            sub_out_b = sub_out_pre_b + (full_out_e - sub_out_pre_b) % gs_out
            sub_out_e = sub_out_pre_e - (sub_out_pre_e - full_out_b) % gs_out

        else:
            inv_st = self.stride_ratio.denominator
            lpg = self.l_pad * gs_out
            rpg = self.r_pad * gs_out
            lwg = self.l_wing_sz * gs_out
            rwg = self.r_wing_sz * gs_out
            full_in_adj_b = full_in_b - lpg
            full_in_adj_e = full_in_e + rpg
            if sub_in_b == full_in_b:
                sub_in_adj_b = full_in_adj_b
            else:
                sub_in_adj_b = sub_in_b - (inv_st - 1) * gs_out
            if sub_in_e == full_in_e:
                sub_in_adj_e = full_in_adj_e
            else:
                sub_in_adj_e = sub_in_e + (inv_st - 1) * gs_out
            if full_in_adj_e - full_in_adj_b < lwg + rwg:
                return empty
            if sub_in_adj_e - sub_in_adj_b < lwg + rwg:
                return empty
            full_out_b = full_in_adj_b + lwg
            full_out_e = full_in_adj_e - rwg
            sub_in_b = sub_in_adj_b + lwg
            sub_in_e = sub_in_adj_e - rwg

        return (full_out_b, full_out_e), (sub_out_b, sub_out_e), gs_out

    def _input_range(self, full_out, sub_out, gs_out):
        """
        Return the full and sub input range in physical coordinates.
        Assume the output ranges full_out and sub_out are in a grid spacing
        of gs_out.
        """
        full_out_b, full_out_e = full_out
        sub_out_b, sub_out_e = sub_out
        gs_in = gs_out / self.stride_ratio
        assert gs_in == int(gs_in)
        gs_in = int(gs_in)
        empty = (-1, -1), (-1, -1), gs_out

        if self.stride_ratio >= 1:
            lwg = self.l_wing_sz * gs_in
            rwg = self.r_wing_sz * gs_in
            lpg = self.l_pad * gs_in
            rpg = self.r_pad * gs_in

            full_in_pre_b = full_out_b - lwg
            full_in_pre_e = full_out_e + rwg
            sub_in_pre_b = sub_out_b - lwg
            sub_in_pre_e = sub_out_e + rwg

            if full_in_pre_e - full_in_pre_b < lwg + rwg:
                return empty
            if sub_in_pre_e - sub_in_pre_b < lwg + rwg:
                return empty

            full_in_b = full_in_pre_b + lpg
            full_in_e = full_in_pre_e - rpg
            sub_in_b = max(sub_in_pre_b, full_in_b)
            sub_in_e = max(sub_in_pre_e, full_in_e)

        else:
            lwg = self.l_wing_sz * gs_out
            rwg = self.r_wing_sz * gs_out
            lpg = self.l_pad * gs_out
            rpg = self.r_pad * gs_out

            full_in_adj_b = full_out_b - lwg
            full_in_adj_e = full_out_e + rwg
            sub_in_adj_b = sub_out_b - lwg
            sub_in_adj_e = sub_out_e + rwg

            full_in_b = full_in_adj_b + lpg
            full_in_e = full_in_adj_e - rpg
            sub_in_b = sub_in_adj_b + (full_in_e - sub_in_adj_b) % gs_in
            sub_in_e = sub_in_adj_e - (sub_in_adj_e - full_in_b) % gs_in

        return (full_in_b, full_in_e), (sub_in_b, sub_in_e), gs_in


    def _output_range_bck(self, in_b, in_e, in_l):
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
        lp, rp = self.l_pad, self.r_pad
        w = lw + rw

        if self.stride_ratio >= 1:
            stride = self.stride_ratio.numerator
            # bounds in P-input
            p_in_b = in_b + int(in_b != 0) * lp
            p_in_e = in_e + lp + int(in_e == in_l) * rp
            p_in_l = in_l + lp + rp

            if p_in_b + lw + rw > p_in_l:
                out_b = -1
            else:
                out_dense_b = p_in_b + lw - lw
                out_b = math.ceil(out_dense_b / stride)
            if p_in_e - lw - rw < 0:
                out_e = -1
            else:
                out_dense_e = p_in_e - rw - lw
                out_e = out_dense_e // stride
            if p_in_l - lw - rw < 0:
                out_l = -1
            else:
                out_dense_l = p_in_l - lw - rw
                out_l = out_dense_l // stride

        else:
            ist = self.stride_ratio.denominator

            # index in SP-input
            sp_in_b = (in_b * ist + int(in_b != 0) * lp - int(in_b != 0) * (ist
                - 1))
            sp_in_e = (in_e * ist + lp + int(in_e == in_l) * rp + int(in_e !=
                in_l) * (ist - 1))
            sp_in_l = in_l * ist + lp + rp

            if sp_in_b + lw + rw > sp_in_l:
                out_b = -1
            else:
                out_b = sp_in_b + lw - lw
            out_e = max(-1, sp_in_e - rw - lw)
            out_l = max(-1, sp_in_l - rw - lw)

        return out_b, out_e, out_l 

def input_range(source, dest, full_out, sub_out, grid_spacing):
    """
    Compute the physical coordinate range of the input corresponding
    to the given full and sub output ranges.  Assume consecutive tensor
    elements are physically grid_spacing units apart.
    """
    vc = dest
    full = full_out[0], full_out[1] - 1
    sub = sub_out[0], sub_out[1] - 1
    gs = grid_spacing
    while True:
        full, sub, gs = vc._input_range(full, sub, gs)
        fmt = 'input_range: full: {}, sub: {}, gs: {}'
        print(fmt.format(full, sub, gs))
        if vc is source:
            break
        vc = vc.parent
    return (full[0], full[1] + 1), (sub[0], sub[1] + 1), gs 


def output_range(source, dest, full_in, sub_in, grid_spacing):
    """
    Compute the physical coordinate range of the output of the chain of
    convolutions source => dest, assuming the given full and sub input ranges.
    Assume pairs of consecutive elements in the input are grid_spacing physical
    distance units apart.
    """
    vc = source
    full = full_in[0], full_in[1] - 1
    sub = sub_in[0], sub_in[1] - 1
    gs = grid_spacing
    while True:
        full, sub, gs = vc._output_range(full, sub, gs)
        fmt = 'output_range: full: {}, sub: {}, gs: {}'
        print(fmt.format(full, sub, gs))
        if vc is dest:
            break
        vc = vc.child
    return (full[0], full[1] + 1), (sub[0], sub[1] + 1), gs 


def to_index(full, sub, gs):
    """
    Given a full range, sub range, and grid spacing, calculate
    the corresponding tensor indices for sub_b, sub_e, full_e
    """
    f0 = full[0]
    return (sub[0] - f0) // gs, (sub[1] - f0) // gs, (full[1] - f0) // gs



def recep_field(source, dest, out_b, out_e, out_len):
    """
    Calculate the input tensor index range [in_b, in_e) receptive field of the
    output tensor range [out_b, out_e), produced by the chain of connected
    transformations from source -> ... -> dest. out_len is the length of the
    actual output.
    Returns in_b, in_e, input_length
    """
    # We need this check because there is no other convenient way to recognize
    # the empty interval.
    if out_b == out_e:
        return 0, 0, 0

    vc = dest
    b, e, l = out_b, out_e - 1, out_len - 1
    while True:
        b_p, e_p, l_p = b, e, l
        b, e, l = vc._recep_field(b, e, l)
        fmt = 'recep_field: in: [{}, {}) of {}, out: [{}, {}) of {}, {}'
        print(fmt.format(b, e + 1, l + 1, b_p, e_p + 1, l_p + 1, vc))
        if vc is source:
            break
        vc = vc.parent
    return b, e + 1, l + 1


def output_range_bck(source, dest, in_b, in_e, in_len):
    """
    Calculates the maximal set of output elements [out_b, out_e) produced by
    the chain of transformations source -> ... -> dest, in which each element
    has a receptive field which is a subset of [in_b, in_e) if in_b == 0 or
    in_e == in_len, the operation is allowed to make use of left or right
    padding, respectively.
    Will return [0, 0) if there are no output elements that satisfy the
    criteria.
    """
    if in_b == in_e:
        return 0, 0, 0

    vc = source
    b, e, l = in_b, in_e - 1, in_len - 1
    while True:
        b_p, e_p, l_p = b, e, l
        b, e, l = vc._output_range(b, e, l)
        fmt = 'output_range: in: [{}, {}) of {}, out: [{}, {}) of {}, {}'
        print(fmt.format(b_p, e_p + 1, l_p + 1, b, e + 1, l + 1, vc))
        if vc is dest:
            break
        vc = vc.child
    return b, e + 1, l + 1


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

