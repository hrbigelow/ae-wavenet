# An instance of this class represents the coordinate relationship between an
# output element and its input receptive field.
class FieldOffset(object):
    '''calculate left and right receptive field offsets
    for various situations'''
    def __init__(self, offsets=None, filter_sz=None, multiplier=1):
        '''Converts a padding strategy into actual padding values.
        multiplier represents '''
        if isinstance(offsets, tuple):
            self.left = offsets[0] * multiplier
            self.right = offsets[1] * multiplier
        elif filter_sz is not None:
            total_offset = filter_sz - 1
            left_tmp = total_offset // 2
            right_tmp = total_offset - left_tmp
            self.left = left_tmp * multiplier
            self.right = right_tmp * multiplier

    def __repr__(self):
        return 'left: {}, right: {}'.format(self.left, self.right)

    def total(self):
        '''The total size of the receptive field of one output element'''
        return self.left + self.right

