# An instance of this class represents the coordinate relationship between an
# output element and its input receptive field.


class FieldOffset(object):
    '''calculate left and right receptive field offsets for various situations.
    self.left_ind and self.right_ind are in tensor index coordinate offsets.
    self.left and self.right are in external coordinates, related by
    field_spacing.
    
    field_spacing is some positive integer, which represents how far apart in
    the external coordinate system are a consecutive pair of output tensor
    elements.
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
        if self.is_downsample:
            self.stride_ratio = 1.0 / self.stride
        else:
            self.stride_ratio = self.stride

        if isinstance(wing_sizes, tuple):
            self.left_wing_sz = wing_sizes[0]
            self.right_wing_sz = wing_sizes[1]
        elif filter_sz is not None:
            total_wing_sz = filter_sz - 1
            self.left_wing_sz = total_wing_sz // 2
            self.right_wing_sz = total_wing_sz - self.left_wing_sz

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
            return self.input_size(this_input_size)

    def _left(self):
        if self.parent is None:
            pos_cumul, sr_cumul = 0, 1 
        else:
            pos_cumul, sr_cumul = self._left()

        left_ind = self.left_wing_sz - self.left_pad
        pos = pos_cumul + left_ind * sr_cumul
        return (pos, sr_cumul * self.stride_ratio)
        
    def left(self):
        '''Considering the full chain of transformations, calculate the number
        of positions the start of the input is to the left of the start of the
        output.'''
        pos, stride_ratio = self._left()
        return pos

    def _right(self):
        if self.parent is None:
            return (0, self.stride_ratio)
        else:
            pos_cumul, sr_cumul = self._right()
            right_ind = self.right_wing_sz - self.right_pad
            pos = pos_cumul - right_ind * sr_cumul
            return (pos, sr_cumul * self.stride_ratio)

    def right(self):
        '''Considering the full chain of transformations, calculate the number
        of positions the end of the input is to the right of the end of the
        output'''
        pos, stride_ratio = self._right()
        return pos

    








