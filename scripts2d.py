# from math import gcd
# from functools import reduce
from scipy.spatial import ConvexHull, convex_hull_plot_2d
# from itertools import combinations
# from scipy import stats

# import csv
import numpy as np
# import matplotlib.pyplot as plt

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

def single_sumset(A, iterations=10):
    m, nA =  run_exps(process(A), iterations)
    print(f'm = {m}')
    print(f'nA = {nA}')

def volume_of_convex_hull(points):
    if np.shape(points)[1] == 1:
        return max(points)[0]
    return ConvexHull(points).volume

def run_exps(curr_set, iterations):
    m = volume_of_convex_hull(curr_set)
    n_set = np.copy(curr_set)
    for _ in range(iterations):
        n_set = sum_sets(n_set, curr_set)
    return m, deprocess(n_set)

single_sumset([(0,0), (0,1), (1,0)], 2)