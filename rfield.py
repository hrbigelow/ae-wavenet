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
    def __init__(self, offsets=None, filter_sz=None, field_spacing=1):
        '''Converts a padding strategy into actual padding values.
        '''
        self.field_spacing = field_spacing

        if isinstance(offsets, tuple):
            self.left_ind = offsets[0]
            self.right_ind = offsets[1]
        elif filter_sz is not None:
            total_offset = filter_sz - 1
            self.left_ind = total_offset // 2
            self.right_ind = total_offset - self.left_ind

        self.left = self.left_ind * self.field_spacing
        self.right = self.right_ind * self.field_spacing

    def __repr__(self):
        return 'left_ind: {}, right_ind: {}, field_spacing: {}'.format(self.left_ind,
                self.right_ind, self.field_spacing)

    def total(self):
        '''The total size of the receptive field of one output element'''
        return self.left_ind + self.right_ind
