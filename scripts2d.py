# from math import gcd
# from functools import reduce
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from collections import Counter
# from itertools import combinations
# from scipy import stats

import csv
import math
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


def intersection(basis, translations, iters):
    rtn = []
    # get all dilations
    basis_dilations = [[(0,0)], basis]
    curr_basis = basis.copy()
    for _ in range(iters):
        curr_basis = sum_sets(curr_basis, basis)
        basis_dilations.append(curr_basis)

    for h in range(iters):
        points_list = set()
        intersections = []
        # i'th dilation
        for i in range(h+1,-1,-1):
            dilation = basis_dilations[i]
            # want all partitions of i. So j*1st trans, and (i-j)*2nd trans
            if len(translations) == 2:
                for j in range(h+1-i, -1, -1):
                    tdilation = [(e[0] + j * translations[0][0], e[1] + j * translations[0][1]) for e in dilation]
                    tdilation = [(e[0] + (h+1-i-j) * translations[1][0], e[1] + (h+1-i-j) * translations[1][1]) for e in tdilation]
                    for e in tdilation:
                        if e in points_list:
                            intersections.append(e)
                    points_list.update(tdilation)
            elif len(translations) == 1:
                tdilation = [(e[0] + (h+1-i) * translations[0][0], e[1] + (h+1-i) * translations[0][1]) for e in dilation]
                for e in tdilation:
                    if e in points_list:
                        intersections.append(e)
                points_list.update(tdilation)
            else:
                assert False
        rtn.append(np.array(intersections))
    return rtn

def second_intersection(first_intersections):
    u, cc = np.unique(first_intersections, axis=0, return_counts=True)
    return u[cc > 1]


def single_sumset(A, iterations=None, slice=None, plot=False, show_intersections=False, basis=None, translations=None):
    if show_intersections is not False:
        assert basis is not None
        assert translations is not None
        assert iterations is not None

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
            if show_intersections:
                intersections = intersection(basis, translations, iterations)
                # print(intersections)

                assert slice[1] <= iterations + 1
                iterations = slice[1] - slice[0]
                a = int(np.ceil(iterations / 2))
                _, axis = plt.subplots(a,2)
                for j in range(iterations):
                    m = 1 if j % 2 == 1 else 0
                    n = j // 2

                    hull = ConvexHull(points_arry[j+slice[0]])
                    points = np.array(points_arry[j+slice[0]])
                    first_intersections = np.array(intersections[j+slice[0]])
                    second_intersections = second_intersection(first_intersections)
                    axis[n,m].plot(points[:,0], points[:,1], 'o', markersize=4)
                    if first_intersections.size != 0:
                        axis[n,m].plot(first_intersections[:,0], first_intersections[:,1], 's', markersize=3)
                        if second_intersections.size != 0:
                            axis[n,m].plot(second_intersections[:,0], second_intersections[:,1], 'v', color='black', markersize=4)
                    for simplex in hull.simplices:
                        axis[n,m].plot(points[simplex, 0], points[simplex, 1], 'k--')
                    axis[n,m].set_title(f'iter = {j+slice[0]}')
                    axis[n,m].grid(color = 'gray', linestyle = '--', linewidth = 0.5)
            else:
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
    points_list = [np.copy(n_set)]
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
            d21, d22 = d12 - d11, d13 - d12
            d31 = d22 - d21

            if d31 == 0:
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
        points_list.append(n_set)
        lengths_arry.append(len(n_set))
        i+=1

    assert k is not None
    mm = m if real_m is None else real_m 

    # Calculate b and c
    b = (lengths_arry[k] - lengths_arry[k-1]) - (mm*((2*(k))+1))
    c = lengths_arry[k-1] - (mm*((k)**2)) - (b*(k))
                
    return  m, mm, b, c, k, deprocess(points_list[-3]), lengths_arry, points_list 


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


def write_to_csv(fname, rows):
    with open(fname, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(('b' , 'c', 'k', 'A'))
        writer.writerows(rows)

def primitive_triangle_basis(max_iterations):
    # generators for SL_2(Z)
    gens = [np.array([[1,1], [1,0]]), np.array([[0,-1], [1,0]])]
    # `word` is a random sequence of 0s and 1s of length <= max_iters
    iters = np.random.randint(1, max_iterations)
    word = np.random.randint(0,2, size=iters)
    # generate
    current = np.identity(2)
    for i in word:
        current = np.matmul(current, gens[i])
    # add third element to make it d+2
    a,b,c,d = current[0,0], current[1,0], current[0,1], current[1,1] 
    return [(0,0), (int(a),int(b)), (int(c),int(d))]


def random_primitive_dPn(n, max_iterations):
    rtn = primitive_triangle_basis(max_iterations) 
    # translate if not in quadrant I
    xs, ys = list(zip(*rtn))
    # vertically first
    if np.min(ys) < 0:
        ymin_arg = np.argmin(ys)
        yoffset = - rtn[ymin_arg][1]
        rtn = [(e[0], e[1] + yoffset) for e in rtn]
    #  then horixontally
    if np.min(xs) < 0:
        xmin_arg = np.argmin(xs)
        xoffset = - rtn[xmin_arg][0]
        rtn = [(e[0] + xoffset, e[1]) for e in rtn]

    basis = rtn.copy()
    translations = []
    max_element_range = max_iterations * np.amax(rtn)
    for _ in range(n-1):
        x = tuple(np.random.randint(0, max_element_range, size=2))
        while x in rtn:
            x = tuple(np.random.randint(0, max_element_range, size=2))
        translations.append(x)
        rtn.append(x)
    return rtn, basis, translations


def random_primitive_dPn_exps(n, maxx, iters):
    results = set()
    for _, A in enumerate([random_primitive_dPn(n, maxx)[0] for i in range(iters)]):
        print(A)
        m, mm, b, c, k, _, lengths_arry, _ =  run_exps(A, None)
        if m != mm:
            continue
        res = (b,c,k,tuple(sorted(A)))
        print(lengths_arry[0])
        results.add(res)
    write_to_csv(f'random_2d_exps/privimite_{2+n}gons_{maxx}_{iters}.csv', results)
    

def has_unique_minimal_elements(A, basis, translations):
    # mininal_elements_bound_factor (change this later)
    # TODO: change m to the bound in paper and make lengths_arry large enough
    m = 25 
    _, _, _, _, _, _, lengths_arry, _ =  run_exps(A, m)
    intersections = intersection(basis, translations, m)
    first_intersections = [np.unique(e, axis=0) for e in intersections]
    second_intersections = [second_intersection(e) for e in intersections]
    # Has unique minimal elements of 1st and 2nd degree iff there is perfect self-similarity
    # between the two degrees of intersection and the original points.
    # x is the number of iterations until we see the first intersection
    x, y = 0, 0
    for i in range(len(first_intersections)):
        if len(first_intersections[i]) != 0:
            x = i+1
            break
    for i in range(len(second_intersections)):
        if len(second_intersections[i]) != 0:
            y = i+1
            break
        
    # print(f'x={x}, y={y}')
    # print(f'first_intersections={first_intersections}')
    # print(f'second_intersections={second_intersections}')
    # print(f'lengths_arry={lengths_arry}')

    # formula that we want say that |intersections[i+x]| = lengths_arry[i]
    # and |second_intersection(intersections[i+y])| = lengths_arry[i]
    unique1 = True
    unique2 = True
    for i in range(len(first_intersections) - x):
        # print(len(first_intersections[i+x]), lengths_arry[i])
        if len(first_intersections[i+x]) != lengths_arry[i]:
            # print(f'unique1 false at i={i+x}')
            unique1 = False
            break
    for i in range(len(second_intersections) - y):
        # print(len(second_intersections[i+y]), lengths_arry[i])
        if len(second_intersections[i+y]) != lengths_arry[i]:
            unique2 = False
            # print(f'unique2 false at i={i+y}')
            break
    return unique1, unique2


def filter_uniques_from_random(iters, basis_depth):
    unique1s = []
    unique2s = []
    for _ in range(iters):
        A, basis, translations = random_primitive_dPn(3, basis_depth)
        unique1, unique2 = has_unique_minimal_elements(A, basis, translations)
        if unique1:
            unique1s.append((A, basis, translations))
        if unique2:
            if not unique1:
                print("Second but not first!")
                print((A, basis, translations))
                assert False
            unique2s.append((A, basis, translations))

    return unique1s, unique2s 


def magnitue_grows_with_linear_term_dP2():
    for i in range(20):
        A, _ = primitive_triangle_basis(6)
        print(f"\ni = {i}")
        for j in range(8):
            print(single_sumset(A + [(j+3, j+3)]))

def area_thing(sets):
    i = 0
    for sett in sets:
        print(i)
        i += 1
        single_sumset(sett[0][:-1])
        single_sumset(sett[0][:-2] + sett[0][-1:])
        single_sumset(sett[0])


"""------------------------WORKPSPACE----------------------------"""


# single_sumset([(0, 2), (1, 0), (1, 1), (9, 8), (11, 11)]) TODO: Why is this not primitive?

# unique on both intersections
# single_sumset([(0, 0), (1, 0), (1, 1), (3, 3)], iterations=6, slice=(0,4), plot=True)
# single_sumset([(0, 0), (1, 0), (1, 1), (0, 4)], iterations=6, slice=(0,4), plot=True) 
# single_sumset([(0, 0), (1, 0), (1, 1), (0, 4), (3,3)], iterations=10, slice=(0,4), plot=True)

# non-unique on second intersection
# single_sumset([(0, 0), (1, 0), (1, 1), (3, 3)], iterations=6, slice=(0,4), plot=True)
# single_sumset([(0, 0), (1, 0), (1, 1), (0, 5)], iterations=6, slice=(0,4), plot=True) 
# single_sumset([(0, 0), (1, 0), (1, 1), (0, 5), (3,3)], iterations=10, slice=(0,4), plot=True)

# non-unique on first intersection
# single_sumset([(0, 0), (1, 0), (1, 1), (4, 3)], iterations=6, slice=(0,4), plot=True)
# single_sumset([(0, 0), (1, 0), (1, 1), (0, 4)], iterations=6, slice=(0,4), plot=True) 
# single_sumset([(0, 0), (1, 0), (1, 1), (0, 4), (4,3)], iterations=10, slice=(0,4), plot=True)

# non-unique on both intersection
# single_sumset([(0, 0), (1, 0), (1, 1), (4, 3)], iterations=6, slice=(0,4), plot=True)
# single_sumset([(0, 0), (1, 0), (1, 1), (0, 5)], iterations=6, slice=(0,4), plot=True) 
# single_sumset([(0, 0), (1, 0), (1, 1), (0, 5), (4,3)], iterations=10, slice=(0,4), plot=True)

# d-simplex doesn't play too nice. But maybe obeys something else. Maybe can fix.
# single_sumset([(0, 0), (1, 0), (1, 1), (2,2), (3, 3)], iterations=6, slice=(0,4), plot=True)
# single_sumset([(0, 0), (1, 0), (1, 1), (2,2), (2, 0)], iterations=6, slice=(0,4), plot=True)
# Note: there are no d-simplices (in d=2 at least) with d+3 elements for which the d+3 rd element
# is inside the interior of the convex hull and not the boundary. Need d+4 elements to insert an
# element of the interior.

# for one, two in [((0,3),(2,1)), ((1,2), (0,3)), ((3,3), (1,2)), ((0,1), (2,2))]:
#     single_sumset(
#         [(0, 0), (1, 0), (1, 1), one, two],
#         iterations=10,
#         slice=(0,6),
#         plot=True,
#         show_intersections=True,
#         basis=[(0,0), (1,0), (1,1)],
#         translations=[one, two]
#         )


# for e in filter_uniques_from_random(100, 4):
#     print(e)


# unique for both
uniques12 = [([(1, 1), (0, 0), (0, 1), (3, 0), (3, 3)], [(1, 1), (0, 0), (0, 1)], [(3, 0), (3, 3)]), # paralellogram
([(1, 1), (0, 0), (0, 1), (2, 2), (2, 0)], [(1, 1), (0, 0), (0, 1)], [(2, 2), (2, 0)]),              # trapezoid with paralell
([(1, 0), (0, 1), (1, 1), (3, 1), (2, 2)], [(1, 0), (0, 1), (1, 1)], [(3, 1), (2, 2)]),              # paralellogram
([(1, 1), (0, 1), (1, 0), (3, 1), (1, 2)], [(1, 1), (0, 1), (1, 0)], [(3, 1), (1, 2)]),              # symmetrical diamond shape
([(1, 1), (0, 0), (0, 1), (3, 1), (2, 2)], [(1, 1), (0, 0), (0, 1)], [(3, 1), (2, 2)]),              # trapezoid with non-parallem
([(0, 1), (0, 0), (1, 1), (3, 1), (3, 2)], [(0, 1), (0, 0), (1, 1)], [(3, 1), (3, 2)])]              # paralellogram

# unique for the first degree intersection
uniques1 = [([(0, 0), (2, 1), (1, 1), (6, 3), (1, 4)], [(0, 0), (2, 1), (1, 1)], [(6, 3), (1, 4)]),
([(1, 1), (0, 1), (1, 0), (3, 2), (2, 3)], [(1, 1), (0, 1), (1, 0)], [(3, 2), (2, 3)]),
([(1, 0), (1, 1), (0, 0), (2, 2), (0, 3)], [(1, 0), (1, 1), (0, 0)], [(2, 2), (0, 3)])]
# + uniques12 since if unique on 2 then unique on 1

#TODO: how to compute the binomial basis polynomial for the uniques1 case. Can it also be done for
# the uniques12 case?

#TODO: confirm or deny the area thing on uniques12
# --> doesn't hold for [(0, 1), (0, 0), (1, 1), (3, 1), (3, 2)]. what are its coefficients? How do they compare to
# the coefficients of other uniques12 that do satisfy area thing

# Seems like for 1st intersection to be unique we need d+1 simplices for which one of the basis
# points is in the interior of the convex hull 
#TODO: what makes the shapes in uniques12 and uniques1 different?

#TODO: what happens for d-simplices?


# print(has_unique_minimal_elements([(0, 1), (0, 0), (1, 1), (3, 1), (3, 2)], [(0, 1), (0, 0), (1, 1)], [(3, 1), (3, 2)]))
single_sumset(
    uniques12[0][0],
    basis=uniques12[0][1],
    translations=uniques12[0][2],
    iterations=10,
    slice=(0,8),
    plot=True,
    show_intersections=True,
    )


