import fractions
import math
import numpy as np

"""
See doc/vconv_notes.txt 
"""

class GridRange(object):
    """
    Defines virtual tensor and subrange, embedded in a global coordinate grid.
    The virtual tensor has elements at positions range(full[0], full[1], gs)
    The subrange has elements at positions range(sub[0], sub[1], gs)
    """
    def __init__(self, full, sub, gs):
        self.full = list(full)
        self.sub = list(sub)
        self.gs = gs

    def sub_length(self):
        return (self.sub[1] - self.sub[0] - 1) // self.gs + 1

    def full_length(self):
        return (self.full[1] - self.full[0] - 1) // self.gs + 1

    def valid(self):
        g = self.gs 
        f = self.full
        s = self.sub
        return (
                f[0] <= s[0] and s[0] < s[1] and s[1] <= f[1]
                and g >= 1
                and f[0] % g == (f[1] - 1) % g
                and s[0] % g == (s[1] - 1) % g
                and f[0] % g == s[0] % g
                )

    def __repr__(self):
        fmt = '[{:8}  [{:8}   {:8})   {:8})  |{:3}|  sub_len: {:8}  full_len: {:8}'
        return fmt.format(
                self.full[0], self.sub[0], self.sub[1], self.full[1], self.gs,
                self.sub_length(), self.full_length()
                )


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
            is_downsample=True, name=None, parent=None):
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
            # print(self)
            raise RuntimeError('Filter wing sizes cannot be less than the respective '
                    'padding')
        # print(self)
        

    def __repr__(self):
        fmt = '[{}^{}, {}/{}, {}--{}, "{}"]'
        return fmt.format(
                self.l_wing_sz, self.r_wing_sz,
                self.stride_ratio.numerator, self.stride_ratio.denominator,
                self.l_pad, self.r_pad, self.name)


    def _output_range(self, full_in, sub_in, gs_in):
        full_in_b, full_in_e = full_in
        sub_in_b, sub_in_e = sub_in
        gs_out = gs_in * self.stride_ratio
        assert gs_out == int(gs_out)
        gs_out = int(gs_out)

        if self.stride_ratio >= 1:
            lpg = self.l_pad * gs_in
            rpg = self.r_pad * gs_in
            lwg = self.l_wing_sz * gs_in
            rwg = self.r_wing_sz * gs_in
            full_in_adj_b = full_in_b - lpg 
            full_in_adj_e = full_in_e + rpg
            if full_in_adj_e - full_in_adj_b < lwg + rwg:
                return None
            if sub_in_e - sub_in_b < lwg + rwg:
                return None
            full_out_b = full_in_adj_b + lwg
            full_out_pre_e = full_in_adj_e - rwg
            full_out_e = full_out_pre_e - (full_out_pre_e - full_out_b) % gs_out
            sub_out_pre_b = sub_in_b + lwg
            sub_out_pre_e = sub_in_e - rwg
            # Due to stride filtering, this adjustment may produce
            # an empty or reverse range
            sub_out_b = sub_out_pre_b + (full_out_e - sub_out_pre_b) % gs_out
            sub_out_e = sub_out_pre_e - (sub_out_pre_e - full_out_b) % gs_out
            if sub_out_e - sub_out_b <= 0:
                return None

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
                return None
            if sub_in_adj_e - sub_in_adj_b < lwg + rwg:
                return None
            full_out_b = full_in_adj_b + lwg
            full_out_e = full_in_adj_e - rwg
            sub_out_b = sub_in_adj_b + lwg
            sub_out_e = sub_in_adj_e - rwg

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
                return None
            if sub_in_pre_e - sub_in_pre_b < lwg + rwg:
                return None

            full_in_b = full_in_pre_b + lpg
            full_in_e = full_in_pre_e - rpg
            sub_in_b = max(sub_in_pre_b, full_in_b)
            sub_in_e = min(sub_in_pre_e, full_in_e)

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
            full_in_pre_e = full_in_adj_e - rpg
            e_mod_adjust = - (full_in_pre_e - full_in_b) % gs_in
            full_in_e = full_in_pre_e + e_mod_adjust

            # Due to input spacing, this range may be empty or reversed
            assert sub_in_adj_b <= full_in_e
            assert full_in_b <= sub_in_adj_e
            assert (full_in_e - full_in_b) % gs_in == 0
            sub_in_b = sub_in_adj_b + (full_in_e - sub_in_adj_b) % gs_in
            sub_in_e = sub_in_adj_e - (sub_in_adj_e - full_in_b) % gs_in
            if sub_in_e - sub_in_b <= 0:
                return None

        #print('{} {} {}'.format((full_in_b, full_in_e), (sub_in_b, sub_in_e),
        #    gs_in))
        return (full_in_b, full_in_e), (sub_in_b, sub_in_e), gs_in

    def _output_offsets(self):
        """
        Simple local calculation of offsets from input to output
        on the left and right.  Only works with zero padding and
        stride 1
        """
        if (self.l_pad != 0 or self.r_pad != 0
                or self.stride_ratio != 1):
            raise RuntimeError(
            'Can only call output_offset with no padding and' +
            'unit stride')
        return self.l_wing_sz, -self.r_wing_sz




def input_range(source, dest, out):
    """
    Compute the physical coordinate range of the input corresponding
    to the given full and sub output ranges.  Assume consecutive tensor
    elements are physically grid_spacing units apart.
    """
    vc = dest
    full = out.full[0], out.full[1] - 1
    sub = out.sub[0], out.sub[1] - 1
    gs = out.gs

    # full = full_out[0], full_out[1] - 1
    # sub = sub_out[0], sub_out[1] - 1
    # gs = grid_spacing
    #results = [((full[0], full[1] + 1), (sub[0], sub[1] + 1), gs)]

    while True:
        res = vc._input_range(full, sub, gs)
        if res is None:
            raise RuntimeError('empty input range')
        else:
            full, sub, gs = res
        #results.append(((full[0], full[1] + 1), (sub[0], sub[1] + 1), gs))
        #fmt = 'input_range: full: {}, sub: {}, gs: {}, ind: {}, vc: {}'
        #print(fmt.format(full, sub, gs, to_index(full, sub, gs), vc))
        if vc is source:
            break
        vc = vc.parent
    # return (full[0], full[1] + 1), (sub[0], sub[1] + 1), gs 
    return GridRange((full[0], full[1] + 1), (sub[0], sub[1] + 1), gs)
    #return results


def output_range(source, dest, gin):
    """
    Compute the physical coordinate range of the output of the chain of
    convolutions source => dest, assuming the given full and sub input ranges.
    Assume pairs of consecutive elements in the input are grid_spacing physical
    distance units apart.

    Raises exception if either is an empty range
    """
    vc = source
    full = gin.full[0], gin.full[1] - 1
    sub = gin.sub[0], gin.sub[1] - 1
    gs = gin.gs
    # full = full_in[0], full_in[1] - 1
    # sub = sub_in[0], sub_in[1] - 1
    # gs = grid_spacing
    #results = [((full[0], full[1] + 1), (sub[0], sub[1] + 1), gs)]

    while True:
        res = vc._output_range(full, sub, gs)
        if res is None:
            raise RuntimeError('empty output range')
        else:
            full, sub, gs = res

        #results.append(((full[0], full[1] + 1), (sub[0], sub[1] + 1), gs))
        #fmt = 'output_range: full: {}, sub: {}, gs: {}, ind: {}, vc: {}'
        #print(fmt.format(full, sub, gs, to_index(full, sub, gs), vc))
        if vc is dest:
            break
        vc = vc.child
    # return (full[0], full[1] + 1), (sub[0], sub[1] + 1), gs 
    return GridRange((full[0], full[1] + 1), (sub[0], sub[1] + 1), gs)
    #return results


def output_offsets(source, dest):
    vc = source
    lo, ro = 0, 0
    while True:
        offsets = vc._output_offsets()
        lo += offsets[0]
        ro += offsets[1]
        if vc is dest:
            break
        vc = vc.child
    return lo, ro


def tensor_slice(ref_gcoord, subrange_gcoord):
    """
    Compute the index slice of the tensor input described by ref_gcoord that is
    specified by subrange_gcoord.  
    """
    rsub = ref_gcoord.sub
    rgs = ref_gcoord.gs
    tsub = subrange_gcoord
    assert rsub[0] <= tsub[0] and tsub[1] <= rsub[1]
    bp = tsub[0] - rsub[0]
    ep = tsub[1] - rsub[0]
    assert bp % rgs == 0 and (ep - 1) % rgs == 0
    return bp // rgs, (ep - 1) // rgs + 1


def max_spacing(source, dest, initial_gs):
    """
    Calculate the maximum grid spacing achieved between source a destination
    """
    gs = initial_gs
    max_gs = gs 
    vc = source
    while True:
        gs *= vc.stride_ratio
        assert gs == int(gs)
        max_gs = max(int(gs), max_gs)
        if vc is dest:
            break
        vc = vc.child
    return max_gs


        


