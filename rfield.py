# An instance of this class represents the coordinate relationship between an
# output element and its input receptive field.


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
        self.stride = stride
        self.parent = parent
        self.is_downsample = is_downsample 
        self.left_pad = padding[0]
        self.right_pad = padding[1]

        # stride_ratio is ratio of output spacing to input spacing
        if self.is_downsample:
            self.stride_ratio = self.stride
        else:
            self.stride_ratio = 1.0 / self.stride

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
        return 'left_wing_sz: {}, right_wing_sz: {}, stride: {}, is_downsample: {}'.format(
                self.left_wing_sz, self.right_wing_sz, self.stride, self.is_downsample)

    def _input_size(self, output_size):
        '''calculate the input_size needed to produce the desired output_size'''
        if self.is_downsample:
            input_size = (output_size - 1) * self.stride + 1 \
                    + self.left_wing_sz + self.right_wing_sz \
                    - self.left_pad - self.right_pad
        else:
            input_size = (self.stride - 1 - self.left_pad \
                    - self.right_pad + output_size \
                    + self.left_wing_sz + self.right_wing_sz) \
                    // self.stride
        return input_size        

    def input_size(self, output_size):
        '''calculate input_size size needed to produce the desired output_size.
        recurses up to parents until the end.'''
        this_input_size = self._input_size(output_size)
        if self.parent is None:
            return this_input_size 
        else:
            return self.parent.input_size(this_input_size)

    def _local_bounds(self):
        '''
        '''
        padded_stride = 1 if self.is_downsample else self.stride_ratio
        l_ind = self.left_wing_sz - self.left_pad
        r_ind = self.right_wing_sz - self.right_pad
        l_off = l_ind * padded_stride
        r_off = r_ind * padded_stride
        return l_off, r_off

    def _bounds(self):
        if self.parent is None:
            l_in_pos, r_in_pos, in_stride = 0, 0, 1 
        else:
            l_in_pos, r_in_pos, in_stride = self.parent._bounds()

        l_off, r_off = self._local_bounds()

        # Accumulate offsets
        l_out_pos = l_in_pos + l_off * in_stride
        r_out_pos = r_in_pos + r_off * in_stride
        out_stride = self.stride_ratio * in_stride

        return l_out_pos, r_out_pos, out_stride 
        
    def bounds(self):
        '''Considering the full chain of transformations, calculate the number
        of positions the output boundaries are inset from the input boundaries.
        '''
        l_pos, r_pos, stride = self._bounds()
        return l_pos, r_pos 

    def _min_stride(self, pre_stride, min_stride):
        '''Traverse the chain of transformations, recording the minimal stride
        seen'''
        cur_stride = pre_stride / self.stride_ratio
        min_stride = min(cur_stride, min_stride)

        if self.parent is None:
            return min_stride
        else:
            return self.parent._min_stride(cur_stride, min_stride)


    def print(self, output_size, lpad_rpad_inpad_data_sym='<>-*'):
        '''print a symbolic stack of units with the given output_size'''
        lpad, rpad, ipad, data = list(lpad_rpad_inpad_data_sym)
        span = self.input_size(output_size)
        min_stride = self._min_stride(1, 1)









