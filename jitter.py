import numpy as np

class Jitter(object):
    """Time-jitter regularization.  With probability [p, (1-2p), p], replace
    element i with element [i-1, i, i+1] respectively.  Disallow a run of 3
    identical elements in the output.  Let p = replacement probability, s =
    "stay probability" = (1-2p).
    
    To prevent three-in-a-rows, P(x_t=0|x_(t-2)=2, x_(t-1)=1) = 0 and is
    renormalized.  Otherwise, all conditional distributions have the same
    shape, [p, (1-2p), p].
    """
    def __init__(self, replace_prob):
        """n_win gives number of 
        """
        super(Jitter, self).__init__()
        p, s = replace_prob, (1 - 2 * replace_prob)
        self.cond2d = np.tile([p, s, p], 9).reshape(3, 3, 3)
        self.cond2d[2][1] = [0, s/(p+s), p/(p+s)]

    def __call__(self, win_size):
        """
        populates a tensor mask to be used for jitter, and sends it to GPU for
        next window
        """
        index = np.ones((win_size + 1), dtype=np.int32)
        for t in range(2, win_size):
            p2 = index[t-2]
            p1 = index[t-1]
            index[t] = np.random.choice([0,1,2], 1, False, self.cond2d[p1][p1])
        index[win_size] = 1
        index += np.arange(-1, win_size)
        return index[:-1]
