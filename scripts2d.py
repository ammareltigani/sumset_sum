# from math import gcd
# from functools import reduce
from scipy.spatial import ConvexHull, convex_hull_plot_2d
# from itertools import combinations
# from scipy import stats

# import csv
import numpy as np
import matplotlib.pyplot as plt


def process(l):
    assert len(l) > 0
    if type(l[0]) == int:
        l = [[e] for e in l]
    return np.array(l)


def deprocess(l):
    d = np.shape(l)[1]
    if d == 1:
        return sorted([e for e in l])
    return sorted([tuple(e) for e in l])


def sum_sets(set1, set2):
    assert np.shape(set1)[1] == np.shape(set2)[1]
    return np.unique(np.array([np.add(e1, e2) for e1 in set1 for e2 in set2]), axis=0)


def single_sumset(A, iterations=None, slice=None, plot=False):
    print(f'A = {A}')
    A = process(A) if type(A) == list else A
    m, mm, b, c, k, nA, lenghts_arry, points_arry =  run_exps(A, iterations)
    print(f'p(x) = ({m}) {mm}x^2 + {b}x + {c}  for all k >= {k}')
    print(f'Lengths Array: {lenghts_arry}\n')
    if plot:
        if iterations is None:
            # plot A and nA
            _, axis = plt.subplots(2)

            hull = ConvexHull(A)
            points = np.array(A)
            axis[0].plot(points[:,0], points[:,1], 'o')
            for simplex in hull.simplices:
                axis[0].plot(points[simplex, 0], points[simplex, 1], 'k-')
            axis[0].set_title("A")

            hull = ConvexHull(nA)
            points = np.array(nA)
            axis[1].plot(points[:,0], points[:,1], 'o')
            for simplex in hull.simplices:
                axis[1].plot(points[simplex, 0], points[simplex, 1], 'k-')
            axis[1].set_title("nA")
        else:
            assert slice[1] <= iterations + 1
            iterations = slice[1] - slice[0]
            a = int(np.ceil(iterations / 2))
            _, axis = plt.subplots(a,2)
            for j in range(iterations):
                m = 1 if j % 2 == 1 else 0
                n = j // 2
                hull = ConvexHull(points_arry[j+slice[0]])
                points = np.array(points_arry[j+slice[0]])
                axis[n,m].plot(points[:,0], points[:,1], 'o', markersize=4)
                for simplex in hull.simplices:
                    axis[n,m].plot(points[simplex, 0], points[simplex, 1], 'k--')
                axis[n,m].set_title(f'iter = {j+slice[0]}')
                axis[n,m].grid(color = 'gray', linestyle = '--', linewidth = 0.5)

                # plt.savefig(f"2d_figures/4_416798/{slice[0]}-{slice[1]}.png", bbox_inches='tight', format="png", dpi=1200)

    
    plt.show()
                

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


def run_exps(curr_set, iterations):
    m = volume_of_convex_hull(curr_set)
    n_set = np.copy(curr_set)
    last_three = [np.copy(n_set)]
    lengths_arry = [len(n_set)]
    k = None
    real_m = None
    i = 0
    once = False
    while True:
        if iterations is not None and i >= iterations+3:
            break

        n_set = sum_sets(n_set, curr_set)
        if i >= 3:
            d11 = lengths_arry[i-2] - lengths_arry[i-3]
            d12 = lengths_arry[i-1] - lengths_arry[i-2]
            d13 = lengths_arry[i] - lengths_arry[i-1]
            d21 = d12 - d11
            d22 = d13 - d12
            # print(d22-d21)
            if d22 - d21 == 0:
                if once:
                    if d21 != 2*m:
                        real_m = d21 / 2
                    if k is None:
                        k = i-3
                    if iterations is None:
                        break
                else:
                    once = True
            else:
                once = False
        last_three.append(n_set)
        lengths_arry.append(len(n_set))
        i+=1

    assert k is not None
    mm = m if real_m is None else real_m 

    # Calculate b and c
    b = (lengths_arry[k] - lengths_arry[k-1]) - (mm*((2*(k))+1))
    c = lengths_arry[k-1] - (mm*((k)**2)) - (b*(k))
                
    return  m, mm, b, c, k, deprocess(last_three[-3]), lengths_arry, last_three 


def same_line(points):
    x0,y0  = points[0]
    points = [ (x,y) for x,y in points if x != x0 or y != y0 ]
    slopes = [ (y-y0)/(x-x0) if x!=x0 else None for x,y in points ]
    return all( s == slopes[0] for s in slopes)


def random_set(m, size):
    points = np.random.randint(0, m, size=(size-1, 2))
    points = np.unique(np.insert(points, 0, [0,0], axis=0), axis=0)
    while same_line(points):
        points = np.random.randint(0, m, size=(size, 2))
        points = np.unique(np.insert(points, 0, [0,0], axis=0), axis=0)
    return points

def random_set_exp():
    #TODO: get random and check if primitive, if yes then proceed to plot, else throw out and
    # try again.
    pass


def random_primitive_triangles():
    generator = np.array([[1,1], [1,0]])
    current = generator.copy()
    for _ in range(15):
        a,b,c,d = current[0,0], current[1,0], current[0,1], current[1,1] 
        single_sumset([(0,0), (a,b), (c,d)], iterations=10, plot=False)
        current = np.matmul(current, generator)


# Carefully Created Experiements

# random_primitive_triangles()
# single_sumset([(0,0), (1,0), (0,1)], iterations=4, plot=True)
# single_sumset([(0,0), (2,0), (0,2), (1,0), (0,1), (1,1)], iterations=4, plot=True)
# single_sumset([(0,0), (3,0), (0,3), (1,0), (0,1)], iterations=4, plot=True)

# single_sumset([(0,0), (4,0), (2,2), (2,1), (1,1), (3,1)], iterations=20, slice=(0,6), plot=True)
single_sumset([(0,0), (1,0), (4,0), (3,1), (1,1), (2,2), (3,0)], iterations=8, slice=(0,6), plot=True)
# single_sumset([(0,0), (4,0), (2,2), (2,1), (3,1)], iterations=20, slice=(0,6), plot=True) # symmetric to the last one
# single_sumset([(0,0), (4,0), (2,2), (1,1), (3,1)], iterations=20, slice=(0,6), plot=True) # half as many points in the core, less dense

# Random Experiements

# single_sumset(random_set(10,5), iterations=None, plot=True)
# single_sumset([(0,0), (4,1), (6,7), (9,8)], iterations=None, slice=(34,38), plot=True) #primitive k >= 36
# single_sumset([(0,0), (3,1), (7,1), (9,4)], iterations=17, slice=(14,18), plot=True) #primitive k>=16
# single_sumset([(0,0), (5,2), (9,2), (9,8)], iterations=25, slice=(20, 26), plot=True) #not primitive (missing factor of 2 in second dimension) k >= 24
# single_sumset([(0,0), (2,2), (2,3), (8,3)], iterations=12, slice=(8, 12), plot=True) #not primitive (missing factor of 2 in first dimension) k >= 6
# single_sumset([(0,0), (5,8), (7,0), (9,3)], iterations=75, slice=(0,8), plot=False) #primitive k >= 75 (takes a while to stabilize)

"""
