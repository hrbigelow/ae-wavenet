from sys import exit
import ast
import argparse
import fractions
import numpy as np
import util
import copy

def _round_up(val, step):
    """Round up to nearest step at phase"""
    return val + (-val) % step

class VirtualConv(object):
    def __init__(self, lw, rw, osp, isp, parent=None, name=None):
        self.lw = lw
        self.rw = rw
        self.osp = osp
        self.isp = isp
        self.parent = parent
        if self.parent is not None:
            self.parent.child = self
        self.child = None
        self.name = name

    def __repr__(self):
        return '({},{},{},{}) : {}'.format(self.lw, self.rw, self.osp, self.isp, self.name)

    def mul(self, factor):
        self.lw *= factor
        self.rw *= factor
        self.isp *= factor
        self.osp *= factor

    def reduce(self):
        pass


    def rfield(self, out_b, out_e, in_l):
        """
        Returns receptive field range [in_b, in_e)
        from output range [out_b, out_e).  in_l is the total length
        of input.
        """
        if out_b == out_e:
            return 0, 0
        if_min = 0
        if_max = (in_l - 1) * self.isp
        b_si_min = max(if_min, out_b * self.osp)
        b_si_max = min(if_max, out_b * self.osp + self.lw + self.rw)
        if b_si_min > b_si_max:
            return 0, 0
        b_ii = _round_up(b_si_min, self.isp) // self.isp

        e_si_min = max(if_min, (out_e - 1) * self.osp)
        e_si_max = min(if_max, (out_e - 1) * self.osp + self.lw + self.rw)
        if e_si_min > e_si_max:
            return 0, 0
        e_ii = e_si_max // self.isp + 1
        return b_ii, e_ii

    def ifield(self, in_b, in_e, in_l):
        """
        Returns induced field range [out_b, out_e)
        from input range [in_b, in_e)
        """
        if in_b == in_e:
            return 0, 0
        if_min = self.lw
        if_max = (in_l - 1) * self.isp - self.rw
        b_si_min = max(if_min, in_b * self.isp - self.rw)
        b_si_max = min(if_max, in_b * self.isp + self.lw)
        if b_si_min > b_si_max:
            return 0, 0
        b_oi = _round_up(b_si_min - self.lw, self.osp) // self.osp

        e_si_min = max(if_min, (in_e - 1) * self.isp - self.rw)
        e_si_max = min(if_max, (in_e - 1) * self.isp + self.lw)
        if e_si_min > e_si_max:
            return 0, 0
        e_oi = (e_si_max - self.lw) // self.osp + 1
        return b_oi, e_oi

    def shadow(self, in_b, in_e, in_l):
        """
        Return the index range [shadow_in_b, shadow_in_e), which is the largest
        range of input that lies underneath the induced output [out_b, out_e).
        "underneath" here is based on the physical position induced by the
        structure of the filter, as defined by the left and right wing sizes.
        """
        out_b, out_e = self.ifield(in_b, in_e, in_l)
        if out_b == out_e:
            return 0, 0
        b_si = self.lw + out_b * self.osp
        b_ii = _round_up(b_si, self.isp) // self.isp
        e_si = self.lw + (out_e - 1) * self.osp
        e_ii = e_si // self.isp + 1
        return b_ii, e_ii

    
def merge_child(vc):
    if vc.child is None:
        raise RuntimeError('Cannot merge vc.  No child node')

    lcm = np.lcm.reduce([vc.osp, vc.child.isp])
    m1 = lcm // vc.osp
    m2 = lcm // vc.child.isp
    n1 = copy.copy(vc)
    n2 = copy.copy(vc.child)
    n1.mul(m1)
    n2.mul(m2)
    n1.lw = n1.lw + n2.lw
    n1.rw = n1.rw + n2.rw
    n1.osp = n2.osp
    n1.parent = None
    n1.child = vc.child.child
    return n1

def merge_range(source, dest):
    if source is dest:
        return copy.copy(source)

    vc = source
    while vc.child is not dest:
        vc = merge_child(vc)

    vc = merge_child(vc)
    return vc


class VConvNode(object):
    def __init__(self, index, position=None):
        self.index = index
        self.position = position 
        self.parents = []
        self.children = []
        self.left = None
        self.right = None
        self.is_output = False

    def __repr__(self):
        return '{}:{}'.format(self.index, self.position)

def pad(vc, input, spacing):
    """Add padding to either side of input, initialized with space between
    each element"""
    pad_input = []
    for i in range(vc.l_pad):
        n = VConvNode(-1)
        n.position = input[0].position - spacing
        pad_input.insert(0, n)
    pad_input.extend(input)
    for i in range(vc.r_pad):
        n = VConvNode(-1)
        n.position = input[-1].position + spacing
        pad_input.append(n)
    return pad_input

def space(input, n_nodes):
    """Add n_nodes spacing elements between each element of input.
    Preserve the original input spacing"""
    if len(input) < 2:
        return
    spaced_input = []
    old_space = input[1].position - input[0].position
    new_space = old_space / (n_nodes + 1)
    for i, n in enumerate(input):
        spaced_input.append(n)
        if i < len(input) - 1:
            for p in range(n_nodes):
                ne = VConvNode(-1)
                ne.position = spaced_input[-1].position + new_space
                spaced_input.append(ne)
    return spaced_input

def init_neighbors(nodes):
    pn = None
    for n in nodes:
        n.left = pn
        pn = n
    pn = None
    for n in reversed(nodes):
        n.right = pn
        pn = n

def build_graph(n_input, source, dest):
    unit = fractions.Fraction(1, 1)
    input = list(map(lambda i: VConvNode(i, unit * i), range(n_input)))
    in_layer = input
    vc = source

    while True:
        #input = pad(vc, input, spacing)
        if vc.isp > 1:
            input = space(input, vc.isp - 1)
        if in_layer is None:
            in_layer = input

        init_neighbors(input)

        step = vc.osp
        w = vc.lw + vc.rw
        output = []
        # locate indices of first and last value elements
        n_input = len(input)
        for i, n in enumerate(input):
            if n.index != -1:
                first_val_index = i
                break

        for i in reversed(range(n_input)):
            n = input[i]
            if n.index != -1:
                last_val_index = i
                break

        for oi, ii in enumerate(range(0, n_input - w, step)):
            result = VConvNode(oi)
            result.parents = input[ii:ii + w + 1]
            result.position = result.parents[vc.lw].position
            # add in pseudo-parents if all parents are fill-values
            #if i < first_val_index:
            #    if all(map(lambda p: p.index == -1, result.parents)):
            #        result.parents.append(input[first_val_index])
            #if i > last_val_index:
            #    if all(map(lambda p: p.index == -1, result.parents)):
            #        result.parents.append(input[last_val_index])
            for p in result.parents: 
                p.children.append(result)
            output.append(result)

        input = output
        if vc is dest:
            out_layer = output
            init_neighbors(out_layer)
            for n in out_layer:
                n.is_output = True
            break
        vc = vc.child
    return in_layer, out_layer


def graph_rfield(out_layer, out_b, out_e):
    # Successively search lower bound receptive field for out_b
    if out_b == out_e:
        return 0, 0

    n = out_layer[out_b]
    assert n.index == out_b

    while len(n.parents) > 0:
        # find first non-filler parent 
        for c in n.parents:
            if c.index != -1:
                n = c
                break
    b = n.index
    n = out_layer[out_e-1]
    while len(n.parents) > 0:
        # find last non-filler parent 
        for c in reversed(n.parents):
            if c.index != -1:
                n = c
                break
    e = n.index
    return b, e + 1


def graph_ifield(in_layer, in_b, in_e):
    """Search up through the graph for field of influence of the input range
    [in_b, in_e)
    """
    if in_b == in_e:
        return 0, 0

    n = in_layer[in_b]
    while not n.is_output:
        # Traverse the layers upwards through first-child links
        # and to the right through layer indices
        while n is not None and len(n.children) == 0:
            n = n.right
        if n is None:
            return 0, 0
        else:
            n = n.children[0]
    b = n.index

    n = in_layer[in_e - 1]
    while not n.is_output:
        # Traverse the layers upwards through first-child links
        # and to the right through layer indices
        while len(n.children) == 0 and n.left is not None:
            n = n.left
        if n is None:
            return 0, 0
        else:
            n = n.children[-1]
    e = n.index
    return b, e + 1


def graph_shadow(in_layer, out_layer, in_b, in_e):
    out_b, out_e = graph_ifield(in_layer, in_b, in_e)
    if out_b == out_e:
        return 0, 0
    # search through the in_layer until the matching position is found
    out_b_pos = out_layer[out_b].position
    out_e_pos = out_layer[out_e-1].position
    positions = list(map(lambda n: n.position, in_layer))
    lb_b = util.greatest_lower_bound(positions, out_b_pos)
    for i in range(lb_b, len(in_layer)):
        n = in_layer[i]
        if n.position >= out_b_pos:
            shadow_b = i
            break
    lb_e = util.greatest_lower_bound(positions, out_e_pos)
    shadow_e = lb_e
    #for i in range(lb_e, len(in_layer)):
    #    n = in_layer[i]
    #    if n.position <= out_e_pos:
    #        shadow_e = i
    #        break
    return shadow_b, shadow_e + 1

def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--n-input', '-n', type=int, metavar='INT',
            help='Number of input elements to the transformations')
    p.add_argument('--model-file', '-f', type=str, metavar='STR',
            help='File with the structure of each transformation')
    p.add_argument('--print-override', '-p', action='store_true', default=False,
            help='If given, print comparisons even if they do not differ')
    return p


def main():

    parser = get_parser()
    opts = parser.parse_args()
        
    n_input = opts.n_input
    
    source = None
    cur_vc = None
    with open(opts.model_file) as fh:
        for line in fh.readlines():
            vals = ast.literal_eval(line)
            vals.insert(4, cur_vc)
            cur_vc = VirtualConv(*tuple(vals))
            if source is None:
                source = cur_vc
    dest = cur_vc

    vc = merge_range(source, dest)
    __, min_input = vc.rfield(0, 1, 100000000)
    if n_input < min_input:
        print('Given n_input {} less than minimum input {} required for any output'.format(
            n_input, min_input))
        exit(1)

    print('Original range: ')
    cur_vc = source
    while True:
        print(cur_vc)
        if cur_vc is dest:
            break
        cur_vc = cur_vc.child
    print('')
    print('Merged range:\n{}'.format(vc))
    print('')

    in_layer, out_layer = build_graph(n_input, source, dest) 


    # Test combinations of intervals
    for in_b in range(0, n_input):
        if in_b % 100 == 0:
            print('ifield start range {}'.format(in_b))
        for in_e in range(in_b + 1, n_input + 1):
            t_out = vc.ifield(in_b, in_e, n_input)
            a_out = graph_ifield(in_layer, in_b, in_e) 
            #if t_out != a_out and t_out[1] != t_out[0] and a_out[1] != a_out[0]:
            if t_out != a_out or opts.print_override:
                print('ifield: in: {}, test: {}, act: {}'.format(
                    (in_b, in_e), t_out, a_out))

    __, n_output = vc.ifield(0, n_input, n_input)
    for out_b in range(0, n_output):
        if out_b % 100 == 0:
            print('rfield start range {}'.format(out_b))
        for out_e in range(out_b, n_output + 1):
            t_in = vc.rfield(out_b, out_e, n_input)
            a_in = graph_rfield(out_layer, out_b, out_e)
            #if t_in != a_in and t_in[0] != t_in[1] and a_in[0] != a_in[1]:
            if t_in != a_in or opts.print_override:
                print('rfield: out: {}, test: {}, act: {}'.format(
                    (out_b, out_e), t_in, a_in))

    for in_b in range(0, n_input):
        if in_b % 10 == 0:
            print('shadow start range {}'.format(in_b))
        for in_e in range(in_b, n_input + 1):
            t_s = vc.shadow(in_b, in_e, n_input)
            a_s = graph_shadow(in_layer, out_layer, in_b, in_e)
            #if t_s != a_s and t_s[0] != t_s[1] and a_s[0] != a_s[1]:
            if t_s != a_s or opts.print_override:
                print('shadow: in: {}, test: {}, act: {}'.format(
                    (in_b, in_e), t_s, a_s))

    print('Finished')

if __name__ == '__main__':
    main()

