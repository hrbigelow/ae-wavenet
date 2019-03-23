def _validate_checkpoint_info(ckpt_dir, ckpt_file_template):
    # Unfortunately, Python doesn't provide a way to hold an open directory
    # handle, so we just check whether the directory path exists and is
    # writable during this call.
    import os
    if not os.access(ckpt_dir, os.R_OK|os.W_OK):
        raise ValueError('Cannot read and write checkpoint directory {}'.format(ckpt_dir))
    # test if ckpt_file_template is valid  
    try:
        test_file = ckpt_file_template.format(1000)
    except IndexError:
        test_file = ''
    # '1000' is 2 longer than '{}'
    if len(test_file) != len(ckpt_file_template) + 2:
        raise ValueError('Checkpoint template "{}" ill-formed. ' 
                '(should have exactly one "{{}}")'.format(ckpt_file_template))
    try:
        test_path = '{}/{}'.format(ckpt_dir, test_file)
        if not os.access(test_path, os.R_OK):
            fp = open(test_path, 'w')
            fp.close()
            os.remove(fp.name)
    except IOError:
        raise ValueError('Cannot create a test checkpoint file {}'.format(test_path))


class CheckpointPath(object):
    def __init__():
        self.dir = None
        self.file_template = None
        self.enabled = False

    def enable(self, _dir, file_template):
        validate_checkpoint_info(_dir, file_template)
        self.dir = _dir
        self.file_template = file_template
        self.enabled = True

    def path(self, step):
        if not self.enabled:
            raise RuntimeError('Must call enable first.')
        return '{}/{}'.format(self.dir, self.file_template.format(step))
