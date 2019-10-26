import vconv

model = [
        # lw, rw, numer_stride, denom_stride, lp, rp 
        (199, 200, 160, True, 0, 0, "MFCC"),
        (1, 1, 1, True, 0, 0, "CRR_0"),
        (1, 1, 1, True, 0, 0, "CRR_1"),
        (1, 2, 2, True, 0, 0, "CRR_2"),
        (1, 1, 1, True, 0, 0, "CRR_3"),
        (1, 1, 1, True, 0, 0, "CRR_4"),
        (0, 0, 1, True, 0, 0, "CRR_5"),
        (0, 0, 1, True, 0, 0, "CRR_6"),
        (0, 0, 1, True, 0, 0, "CRR_7"),
        (0, 0, 1, True, 0, 0, "CRR_7"),
        (1, 1, 1, True, 0, 0, "LC_Conv"),
        (12, 12, 5, False, 4, 4, "Upsampling_0"),
        (7, 8, 4, False, 3, 3, "Upsampling_1"),
        (7, 8, 4, False, 3, 3, "Upsampling_2"),
        (7, 8, 4, False, 3, 3, "Upsampling_3"),
        (1, 0, 1, True, 0, 0, "GRCC_0,0"),
        (2, 0, 1, True, 0, 0, "GRCC_0,1"),
        (4, 0, 1, True, 0, 0, "GRCC_0,2"),
        (8, 0, 1, True, 0, 0, "GRCC_0,3"),
        (16, 0, 1, True, 0, 0, "GRCC_0,4"),
        (32, 0, 1, True, 0, 0, "GRCC_0,5"),
        (64, 0, 1, True, 0, 0, "GRCC_0,6"),
        (128, 0, 1, True, 0, 0, "GRCC_0,7"),
        (256, 0, 1, True, 0, 0, "GRCC_0,8"),
        (512, 0, 1, True, 0, 0, "GRCC_0,9"),
        (1, 0, 1, True, 0, 0, "GRCC_1,0"),
        (2, 0, 1, True, 0, 0, "GRCC_1,1"),
        (4, 0, 1, True, 0, 0, "GRCC_1,2"),
        (8, 0, 1, True, 0, 0, "GRCC_1,3"),
        (16, 0, 1, True, 0, 0, "GRCC_1,4"),
        (32, 0, 1, True, 0, 0, "GRCC_1,5"),
        (64, 0, 1, True, 0, 0, "GRCC_1,6"),
        (128, 0, 1, True, 0, 0, "GRCC_1,7"),
        (256, 0, 1, True, 0, 0, "GRCC_1,8"),
        (512, 0, 1, True, 0, 0, "GRCC_1,9")
        ]


p = None
mods = []
for m in model:
    t = (m[0], m[1]), (m[4], m[5]), m[2], m[3], p, m[6] 
    p = vconv.VirtualConv(*t) 
    mods.append(p)

mfcc_vc = mods[0]
last_grcc_vc = mods[-1]

f, s, gs = vconv.output_range(mfcc_vc, last_grcc_vc, (0, 1520), (250, 790), 1)

