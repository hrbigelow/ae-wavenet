class FieldOffset(object):
    '''calculate left and right receptive field offsets
    for various situations'''
    def __init__(self, offsets=None, filter_sz=None):
        '''Converts a padding strategy into actual padding values'''
        if isinstance(offsets, tuple):
            self.left, self.right = offsets 
        elif filter_sz is not None:
            total_offset = filter_sz - 1
            self.left = total_offset // 2
            self.right = total_offset - self.left

