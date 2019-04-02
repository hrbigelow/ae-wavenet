# An instance of this class represents the coordinate relationship between an
# output element and its input receptive field.
import fractions 
import numpy as np

class _Stats(object):
    '''Describes a 1D tensor of positioned elements''' 
    def __init__(self, l_pad, r_pad, size, spc, vspc, l_pos, r_pos):
        self.l_pad = l_pad
        self.r_pad = r_pad
        self.size = size
        self.spc = spc
        self.vspc = vspc
        self.l_pos = l_pos
        self.r_pos = r_pos

    def __repr__(self):
        return 'l_pad: {}, r_pad: {}, size: {}, spc: {}, vspc: {}, l_pos:' \
                ' {}, r_pos: {}'.format(self.l_pad, self.r_pad, self.size, self.spc,
                        self.vspc, self.l_pos, self.r_pos)

class FieldOffset(object):
    '''
    Use this class as a member in individual Convolution or Transpose
    convolution modules.  Allows convenient back-calculation of needed input
    size for a desired output size.  Also, allows calculating the left and
    right offsets of the output from the input across the whole chain of
    transformations.

    '''
    def __init__(self, padding=(0, 0), wing_sizes=None, filter_sz=None,
            stride=1, is_downsample=True, parent=None):
        '''Converts a padding strategy into actual padding values.
        '''
        self.parent = parent
        self.l_pad = padding[0]
        self.r_pad = padding[1]

        # stride_ratio is ratio of output spacing to input spacing
        if is_downsample:
            self.stride_ratio = fractions.Fraction(stride, 1)
        else:
            self.stride_ratio = fractions.Fraction(1, stride)

        if isinstance(wing_sizes, tuple):
            self.left_wing_sz = wing_sizes[0]
            self.right_wing_sz = wing_sizes[1]
        elif filter_sz is not None:
            total_wing_sz = filter_sz - 1
            self.left_wing_sz = total_wing_sz // 2
            self.right_wing_sz = total_wing_sz - self.left_wing_sz
        else:
            raise RuntimeError('Must be called with either '
            'wing_sizes tuple or filter_sz (integer)')

    def __repr__(self):
        return 'left_wing_sz: {}, right_wing_sz: {}, stride_ratio: {}'.format(
                self.left_wing_sz, self.right_wing_sz, self.stride_ratio)

    def _input_size(self, output_size):
        '''calculate the input_size needed to produce the desired output_size'''
        const = self.left_wing_sz + self.right_wing_sz - self.l_pad - self.r_pad
        if self.stride_ratio.denominator == 1:
            input_size = (output_size - 1) * self.stride_ratio.numerator + 1 + const
        else:
            input_size = (self.stride_ratio.denominator + output_size - 1 + const) \
                    // self.stride_ratio.denominator
        return input_size        

    def _input_stride(self, output_stride_r):
        return output_stride_r / self.stride_ratio

    def _input_spacing(self, out_spc, out_vspc):
        if self.stride_ratio > 1:
            in_spc = out_vspc / self.stride_ratio
            in_vspc = in_spc
        else:
            in_spc = out_vspc
            in_vspc = in_spc / self.stride_ratio
        return in_spc, in_vspc

    def input_size(self, output_size):
        '''calculate input_size size needed to produce the desired output_size.
        recurses up to parents until the end.'''
        this_input_size = self._input_size(output_size)
        if self.parent is None:
            return this_input_size 
        else:
            return self.parent.input_size(this_input_size)

    def _local_bounds(self):
        '''For this transformation, calculates the offset between the first elements
        of the input and the output, assuming output element spacing of 1.
        Return values must be adjusted by the actual output element spacing.
        '''
        # spacing is the distance between any two consecutive elements in this
        # tensor, including padding elements 
        #input_spacing = max(fractions.Fraction(1, 1), self.stride_ratio)
        input_spacing = 1
        l_ind = self.left_wing_sz - self.l_pad
        r_ind = self.right_wing_sz - self.r_pad
        l_off = l_ind * input_spacing 
        r_off = r_ind * input_spacing 
        return l_off, r_off

    def _geometry(self, out_size, out_spc, out_vspc, l_out_pos, r_out_pos, accu=None):
        l_off, r_off = self._local_bounds()
        in_size = self._input_size(out_size)
        in_spc, in_vspc = self._input_spacing(out_spc, out_vspc)
        l_in_pos = l_out_pos + l_off * in_spc
        r_in_pos = r_out_pos + r_off * in_spc

        if accu is not None:
            assert isinstance(accu, list)
            if len(accu) == 0:
                first_stats = _Stats(0, 0, out_size, out_spc, out_vspc,
                        l_out_pos, r_out_pos)
                accu.append(first_stats)

            stats = _Stats(self.l_pad, self.r_pad, in_size, in_spc,
                    in_vspc, l_in_pos, r_in_pos)
            accu.append(stats)

        if self.parent is None:
            if accu is None:
                return in_size, l_in_pos, r_in_pos
            else:
                return accu
        else:
            return self.parent._geometry(in_size, in_spc, in_vspc, l_in_pos, r_in_pos, accu)

    def geometry(self, out_size):
        '''calculate in_size, left_pos, right_pos needed for the chain of transformations
        to produce the desired out_size'''
        return self._geometry(out_size, 1, 1, 0, 0, None)

    def print(self, output_size, lpad_rpad_inpad_data_sym='<>-*'):
        '''pretty print a symbolic stack of units with the given output_size'''

        lpad, rpad, ipad, data = list(lpad_rpad_inpad_data_sym)
        stats = self._geometry(output_size, 1, 1, 0, 0, [])

        lcm_denom = fractions.Fraction(np.lcm.reduce(tuple(s.spc.denominator for s in stats)))
        for s in stats:
            s.spc *= lcm_denom
            s.vspc *= lcm_denom
            s.l_pos *= lcm_denom
            s.r_pos *= lcm_denom


        def _dilate_str(string, space_sym, spacing):
            space_str = space_sym * (spacing - 1)
            return string[0] + ''.join(space_str + s for s in string[1:])

        max_l_pos = max(s.l_pos for s in stats)
        max_r_pos = max(s.r_pos for s in stats)

        for st in stats:
            indent = max_l_pos - st.l_pos
            dedent = max_r_pos - st.r_pos

            assert indent == round(indent) 
            assert st.spc == round(st.spc)
            assert st.vspc == round(st.vspc)
            assert st.r_pos == round(st.r_pos)
            core = _dilate_str(data * st.size, ipad, round(st.vspc / st.spc))
            body = lpad * st.l_pad + core + rpad * st.r_pad
            body_dil = _dilate_str(body, ' ', round(st.spc))

            s = ' ' * round(indent) + body_dil + ' ' * round(dedent) + '|'
            print(s)
        return stats

