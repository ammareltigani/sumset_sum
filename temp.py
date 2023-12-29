from scipy.spatial import ConvexHull
import numpy as np

def round_volume(vol):
    orig_vol = vol
    for i in range(12):
        vol = orig_vol * (10**i) 
        if np.abs(np.round(vol) - vol)< 1e-8:
            return np.round(vol) / (10**i)
    return orig_vol

def volume_of_convex_hull(points):
    if np.shape(points)[1] == 1:
        return max(points)[0]
    return round_volume(ConvexHull(points).volume)


B = [(0, 0), (-1, 0), (0, -1), (12, 13), (1, 3)] 
maxx = 6 
heights = [11/6, 8/6, 5/6]
for i in range(maxx+1):
    for a in range(-i, i+1):
        for b in range(-i, i+1):
            for c in range(-i, i+1):
                for d in range(-i, i+1):
                    for e in range(-i, i+1):
                        A = [(B[0][0], B[0][1], a), (B[1][0], B[1][1], b),
                             (B[2][0], B[2][1], c), (B[3][0], B[3][1], d),
                             (B[4][0], B[4][1], e)]
                        try:
                            vol = volume_of_convex_hull(A)
                        except:
                            continue

                        for value in heights:
                            if np.isclose(vol, value):
                                print(f'{value}: {A}')