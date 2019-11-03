import vconv
from enum import Enum
from collections import Counter
from fractions import Fraction
from collections import namedtuple
import itertools

TestInput = namedtuple('TestInput',
        [
            'name', 'lw', 'rw', 'lp', 'rp', 'start', 'l1',
            'l2', 'l3', 'gs', 'strides', 'inv_strides',
            'report_freq'
            ]
        )


t1 = TestInput(
        name='Many Convolutions',
        lw=range(0, 20),
        rw=range(0, 20),
        lp=range(0, 8),
        rp=range(0, 8),
        start=range(0, 1),
        l1=range(25, 26),
        l2=range(25, 26),
        l3=range(50, 51),
        gs=range(1, 2),
        strides=[1,2,3,4,5],
        inv_strides=[2,3,4,5],
        report_freq=10000
        )

skip = 5

t2 = TestInput(
        name='Many inputs',
        lw=range(3, 4),
        rw=range(3, 4),
        lp=range(0, 1),
        rp=range(0, 1),
        start=range(0, 200, skip),
        l1=range(0, 200, skip),
        l2=range(0, 200, skip),
        l3=range(0, 200, skip),
        gs=range(1, 10),
        strides=[1,2,3,4,5],
        inv_strides=[2,3,4,5],
        report_freq=10000
        )

class Result(Enum):
    NO_OUTPUT = 1
    NO_INPUT = 2
    UNEQUAL = 3
    SUCCESS = 4

model = [
        # lw, rw, numer_stride, denom_stride, lp, rp 
        ((199, 200), (0, 0), 160, True, "MFCC"),
        ((1, 1), (0, 0), 1, True, "CRR_0"),
        ((1, 1), (0, 0), 1, True, "CRR_1"),
        ((1, 2), (0, 0), 2, True, "CRR_2"),
        ((1, 1), (0, 0), 1, True, "CRR_3"),
        ((1, 1), (0, 0), 1, True, "CRR_4"),
        ((0, 0), (0, 0), 1, True, "CRR_5"),
        ((0, 0), (0, 0), 1, True, "CRR_6"),
        ((0, 0), (0, 0), 1, True, "CRR_7"),
        ((0, 0), (0, 0), 1, True, "CRR_7"),
        ((1, 1), (0, 0), 1, True, "LC_Conv"),
        ((12, 12), (4, 4), 5, False, "Upsampling_0"),
        ((7, 8), (3, 3), 4, False, "Upsampling_1"),
        ((7, 8), (3, 3), 4, False, "Upsampling_2"),
        ((7, 8), (3, 3), 4, False, "Upsampling_3"),
        ((1, 0), (0, 0), 1, True, "GRCC_0,0"),
        ((2, 0), (0, 0), 1, True, "GRCC_0,1"),
        ((4, 0), (0, 0), 1, True, "GRCC_0,2"),
        ((8, 0), (0, 0), 1, True, "GRCC_0,3"),
        ((16, 0), (0, 0), 1, True, "GRCC_0,4"),
        ((32, 0), (0, 0), 1, True, "GRCC_0,5"),
        ((64, 0), (0, 0), 1, True, "GRCC_0,6"),
        ((128, 0), (0, 0), 1, True, "GRCC_0,7"),
        ((256, 0), (0, 0), 1, True, "GRCC_0,8"),
        ((512, 0), (0, 0), 1, True, "GRCC_0,9"),
        ((1, 0), (0, 0), 1, True, "GRCC_1,0"),
        ((2, 0), (0, 0), 1, True, "GRCC_1,1"),
        ((4, 0), (0, 0), 1, True, "GRCC_1,2"),
        ((8, 0), (0, 0), 1, True, "GRCC_1,3"),
        ((16, 0), (0, 0), 1, True, "GRCC_1,4"),
        ((32, 0), (0, 0), 1, True, "GRCC_1,5"),
        ((64, 0), (0, 0), 1, True, "GRCC_1,6"),
        ((128, 0), (0, 0), 1, True, "GRCC_1,7"),
        ((256, 0), (0, 0), 1, True, "GRCC_1,8"),
        ((512, 0), (0, 0), 1, True, "GRCC_1,9")
        ]

vc = None
vcs = {}
for m in model:
    vc = vconv.VirtualConv(*m, parent=vc)
    vcs[vc.name] = vc

del(vc)



def same_or_upsample_test(vc, x):
    try:
        y = vconv.output_range(vc, vc, x)
    except RuntimeError:
        return Result.NO_OUTPUT
    try:
        xn = vconv.input_range(vc, vc, y)
    except RuntimeError:
        return Result.NO_INPUT

    if xn != x:
        return Result.UNEQUAL
    else:
        return Result.SUCCESS


def downsample_test(vc, x):
    try:
        y = vconv.output_range(vc, vc, x)
    except RuntimeError:
        return Result.NO_OUTPUT
    try:
        xn = vconv.input_range(vc, vc, y)
    except RuntimeError:
        return Result.NO_INPUT

    try:
        yt = vconv.output_range(vc, vc, xn)
    except RuntimeError:
        return Result.NO_OUTPUT
    try:
        xt = vconv.input_range(vc, vc, yt)
    except RuntimeError:
        return Result.NO_INPUT

    if xn != xt:
        return Result.UNEQUAL
    else:
        return Result.SUCCESS


def grid_range(f_b, l1, l2, l3, gs, inv_stride):
    gs *= inv_stride
    s_b = f_b + l1 * gs
    s_e = s_b + l2 * gs + 1
    f_e = s_e + l3 * gs
    return vconv.GridRange((f_b, f_e), (s_b, s_e), gs)


def input_gen(t):
    for lw, rw, lp, rp in itertools.product(t.lw, t.rw, t.lp, t.rp):
        for st in t.strides:
            try:
                vc = vconv.VirtualConv((lw, rw), (lp, rp), st, True, 'Conv', None)
            except RuntimeError:
                continue
            print('lw: {}, rw: {}, lp: {}, rp: {}, st: {}'.format(lw, rw, lp,
                rp, st))
            for spec in itertools.product(t.start, t.l1, t.l2, t.l3, t.gs): 
                yield vc, grid_range(*spec, 1)
        for ist in t.inv_strides:
            try:
                vc = vconv.VirtualConv((lw, rw), (lp, rp), ist, False, 'Conv', None)
            except RuntimeError:
                continue
            print('lw: {}, rw: {}, lp: {}, rp: {}, ist: {}'.format(lw, rw, lp,
                rp, ist))
            for spec in itertools.product(t.start, t.l1, t.l2, t.l3, t.gs): 
                yield vc, grid_range(*spec, vc.stride_ratio.denominator)


def main_test(inputs):
    t = inputs
    c = 0
    results = Counter() 
    print('Test: {}'.format(t.name))
    for vc, x in input_gen(t): 
        if vc.stride_ratio.numerator > 1: 
            res = downsample_test(vc, x)
        else:
            res = same_or_upsample_test(vc, x)
        results[res] += 1
        if c > 0 and c % t.report_freq == 0:
            print(results)
        c += 1

    print('Finished')
    print('Results: {}'.format(results))


x = vconv.GridRange((0, 250000), (0, 250000), 1)
y = vconv.output_range(vcs['MFCC'], vcs['GRCC_1,9'], x)
xi = vconv.input_range(vcs['MFCC'], vcs['GRCC_1,9'], y)

#print('x0: {}'.format(x))
#print('y0: {}'.format(y))
#print('xi: {}'.format(xi))


def autoenc_test(vcs, in_len, slice_beg):
    enc = vcs['MFCC'], vcs['Upsampling_3']
    dec = vcs['GRCC_0,0'], vcs['GRCC_1,9']
    mfcc = vcs['MFCC'], vcs['MFCC']
    autoenc = vcs['MFCC'], vcs['GRCC_1,9']

    full_in = vconv.GridRange((0, in_len), (0, in_len), 1)
    full_mfcc = vconv.output_range(*mfcc, full_in)
    full_out = vconv.output_range(*autoenc, full_in)

    out_req = vconv.GridRange(full_out.full, (slice_beg, slice_beg + 100), 1)
    mid_req = vconv.input_range(*dec, out_req)
    in_req = vconv.input_range(*enc, mid_req)
    in_act = in_req
    mfcc_act = vconv.output_range(*mfcc, in_act)
    mid_act = vconv.output_range(*enc, in_act)

    # wav -> wav_mid 
    wav_mid_sl = vconv.tensor_slice(in_act, mid_req.sub)
    # wav_mid_ten = wav_ten[wav_mid_sl]

    # lcond -> lcond_sl
    lcond_sl = vconv.tensor_slice(mid_act, mid_req.sub)
    # lcond_sl_ten = lcond_ten[lcond_sl]
    
    # wav -> wav_out 
    # +1 since it is predicting the next step
    wav_out_sl = vconv.tensor_slice(in_act, out_req.sub)
    # wav_out_ten = wav_ten[sl_b+1:sl_e+1]

    mfcc_in_sl = vconv.tensor_slice(full_mfcc, mfcc_act.sub)

    print('{:10}: {}'.format('full_in', full_in))
    print('{:10}: {}'.format('full_mfcc', full_mfcc))
    print('{:10}: {}'.format('in_req', in_req))
    print('{:10}: {}'.format('mfcc_req', mfcc_act))
    print('{:10}: {}'.format('mid_req', mid_req))
    print('{:10}: {}'.format('mid_act', mid_act))
    print('{:10}: {}'.format('out_req', out_req))
    print('{:10}: {}'.format('full_out', full_out))

    print('wav_mid_sl: {}  len: {}'.format(wav_mid_sl, wav_mid_sl[1] -
        wav_mid_sl[0]))
    print('mfcc_in_sl: {}  len: {}'.format(mfcc_in_sl, mfcc_in_sl[1] -
        mfcc_in_sl[0]))
    print('lcond_sl: {}  len: {}'.format(lcond_sl, lcond_sl[1] - lcond_sl[0]))
    print('wav_out_sl: {}  len: {}'.format(wav_out_sl, wav_out_sl[1] - wav_out_sl[0]))


encoder = vcs['MFCC'], vcs['LC_Conv']
encoder_clip = encoder[0].child, encoder[1]
upsample = vcs['Upsampling_0'], vcs['Upsampling_3']
half_upsample = vcs['Upsampling_2'], vcs['Upsampling_3']
decoder = vcs['GRCC_0,0'], vcs['GRCC_1,9']
autoenc_clip = encoder[0].child, decoder[1] 

def phase_test(vc_range, n_sub_win, winsize):
    c = Counter()
    for b in range(n_sub_win):
        out = vconv.GridRange((0, 90000), (b, b + winsize), 1)
        input = vconv.input_range(*vc_range, out)
        c[input.sub_length()] += 1
        # print(mfcc.sub_length(), end=' ')
    print(c)


#print('Phase test for autoencoder')
#phase_test(autoenc_clip, 100)

print('Phase test for upsample')
phase_test(upsample, 20, 2146)
print()

print('Phase test for half upsample')
phase_test(half_upsample, 20, 2146)
print()

print('Phase test for encoder_clip + upsample')
phase_test((encoder_clip[0], upsample[1]), 6000, 2146)
print()

print('Phase test for decoder')
phase_test(decoder, 6000, 100)
print()


def usage_test(vc_range, winsize):
    c = Counter()
    for b in range(winsize):
        out = vconv.GridRange((0, 100000), (b, b + 1), 1)
        input = vconv.input_range(*vc_range, out)
        slice = vconv.tensor_slice(input, input.sub)
        c[slice] += 1
    print(c)

winsize = 10000
print('Usage test for window size {}'.format(winsize))
usage_test((upsample[0], decoder[1]), winsize)

# for s in range(56730, 57073, 30):
#     autoenc_test(vcs, 100000, s)


#for t in (t2, t1):
#    main_test(t)



#vc = mfcc_vc
#while vc.child is not None:
#    f, s = (0, 1000), (150, 850) 
#    forward = vconv.output_range(vc, vc, f, s, gs)
#    f, s, gs = forward[-1]
#    backward = vconv.input_range(vc, vc, f, s, gs)
#    print('f_in: {}, f_out: {}, {}'.format(forward[0][0], forward[1][0], vc))
#    print('b_in: {}, b_out: {}, {}'.format(backward[1][0], backward[0][0], vc))
#    vc = vc.child
#    print("")


