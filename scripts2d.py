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

def single_sumset(A, max_iterations=10, plot=False):
    m, b, c, nA, k =  run_exps(process(A), max_iterations)
    print(f'p(x) = {m}x^2 + {b}x + {c}  for all k >= {k}')
    if plot:
        hull = ConvexHull(nA)
        points = np.array(nA)
        plt.plot(points[:,0], points[:,1], 'o')
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
        plt.show()

def volume_of_convex_hull(points):
    if np.shape(points)[1] == 1:
        return max(points)[0]
    return ConvexHull(points).volume

def run_exps(curr_set, max_iterations):
    m = volume_of_convex_hull(curr_set)
    n_set = np.copy(curr_set)
    last_three = [np.copy(n_set)]
    lengths_arry = [len(n_set)]
    k = None
    for i in range(max_iterations):
        n_set = sum_sets(n_set, curr_set)
        if i >= 3:
            diff1 = lengths_arry[i-2] - lengths_arry[i-3]
            diff2 = lengths_arry[i-1] - lengths_arry[i-2]
            diff3 = lengths_arry[i] - lengths_arry[i-1]
            if diff2 - diff1 == diff3 - diff2 == 2*m:
                k = i-3
                break
            last_three.pop(0)
            last_three.append(n_set)
        else:
            last_three.append(n_set)
        lengths_arry.append(len(n_set))

    
    assert k is not None
    # Calculate b and c
    b = (lengths_arry[k+1] - lengths_arry[k]) - (m*((2*(k+1))+1))
    c = lengths_arry[k] - (m*((k+1)**2)) - (b*(k+1))
                
    return m, b, c, deprocess(last_three[0]), k+1 

# single_sumset([(0,0), (0,3), (3,4), (5,17)], max_iterations=50, plot=True)
single_sumset([(0,0), (1,0), (0,1), (1,1)], max_iterations=50, plot=True)