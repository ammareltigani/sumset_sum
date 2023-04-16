from scipy.spatial import ConvexHull
from sympy import *

import csv
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

def intersections(basis, translations, iters):
    rtn = []
    # get all dilations
    basis_dilations = [[(0,0)], basis]
    curr_basis = basis.copy()
    for _ in range(iters):
        curr_basis = sum_sets(curr_basis, basis)
        basis_dilations.append(curr_basis)

    for h in range(iters):
        points_list = []
        # i'th dilation
        for i in range(h+1,-1,-1):
            dilation = basis_dilations[i]
            # want all partitions of i. So j*1st trans, and (i-j)*2nd trans
            if len(translations) == 2:
                for j in range(h+1-i, -1, -1):
                    tdilation = [(e[0] + j * translations[0][0], e[1] + j * translations[0][1]) for e in dilation]
                    tdilation = [(e[0] + (h+1-i-j) * translations[1][0], e[1] + (h+1-i-j) * translations[1][1]) for e in tdilation]
                    points_list.extend(tdilation)
            elif len(translations) == 1:
                tdilation = [(e[0] + (h+1-i) * translations[0][0], e[1] + (h+1-i) * translations[0][1]) for e in dilation]
                points_list.extend(tdilation)
            else:
                # Not implemented yet for d+4
                assert False

        unique, counts = np.unique(np.array(points_list), axis=0, return_counts=True)
        l = max(iters, max(counts))
        all_intersections = [[] for _ in range(l)]
        for j in range(l):
            all_intersections[j] = unique[counts > j]
        
        rtn.append(all_intersections)

    return rtn


def single_sumset(A, iterations, slice=None, plot=False, show_intersections=False, basis=None, translations=None):
    if show_intersections is not False:
        assert basis is not None
        assert translations is not None
        assert iterations is not None

    print(f'A = {A}')
    A = process(A) if type(A) == list else A
    m, mm, b, c, k, _, lenghts_arry, points_arry =  run_exps(A, iterations)
    print(f'p(x) = ({m}) {mm}x^2 + {b}x + {c}  for all k >= {k}')
    print(f'Lengths Array: {lenghts_arry}\n')

    if plot:
        assert slice is not None
        if show_intersections:
            inters = intersections(basis, translations, iterations)
            plot_sumset(
                iterations,
                slice,
                inters=inters,
            )
        else:
            plot_sumset(
                iterations,
                slice,
                points_arry=points_arry,
            )
            


def plot_sumset(iterations, slice, points_arry=None, inters=None):
    assert points_arry is not None or inters is not None
    assert points_arry is None or inters is None
    assert slice[1] <= iterations + 1

    if inters is not None:
        points_arry = [e[0] for e in inters]

    iterations = slice[1] - slice[0]
    a = int(np.ceil(iterations / 2))
    _, axis = plt.subplots(a,2)
    for j in range(iterations):
        m = 1 if j % 2 == 1 else 0
        n = j // 2

        hull = ConvexHull(points_arry[j+slice[0]])
        points = np.array(points_arry[j+slice[0]])
        axis[n,m].plot(points[:,0], points[:,1], 'o', markersize=4)

        if inters is not None:
            colors = ['yellow', 'orange', 'red', 'black']
            for i in range(len(colors)):
                ith_intersection = np.array(inters[j+slice[0]][i+1])
                if ith_intersection.size != 0:
                    axis[n,m].plot(ith_intersection[:,0], ith_intersection[:,1], 'o', color=colors[i], markersize=4)

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


def run_exps(curr_set, min_iterations):
    m = volume_of_convex_hull(curr_set)
    n_set = np.copy(curr_set)
    points_list = [np.copy(n_set)]
    lengths_arry = [len(n_set)]
    k = None
    real_m = None
    i = 0
    once = False
    while k is None or i  < min_iterations:
        n_set = sum_sets(n_set, curr_set)
        if i >= 3 and k is None:
            d11 = lengths_arry[i-2] - lengths_arry[i-3]
            d12 = lengths_arry[i-1] - lengths_arry[i-2]
            d13 = lengths_arry[i] - lengths_arry[i-1]
            d21, d22 = d12 - d11, d13 - d12
            d31 = d22 - d21

            if d31 == 0:
                if once:
                    if d21 != 2*m:
                        real_m = d21 / 2
                    k = i-3
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


def write_to_csv(fname, rows, legend=('b' , 'c', 'k', 'A')):
    with open(fname, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(legend)
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
    max_element_range = 10
    for _ in range(n-1):
        x = tuple(np.random.randint(0, max_element_range, size=2))
        while x in rtn:
            x = tuple(np.random.randint(0, max_element_range, size=2))
        translations.append(x)
        rtn.append(x)
    return rtn, basis, translations


def random_primitive_dPn_exps(n, maxx, maxk, iters):
    results = set()
    for _, basis, translations in [random_primitive_dPn(n, maxx) for _ in range(iters)]:
        m, mm, b, c, k, _, _, _ =  run_exps(basis + translations, maxk)
        if m != mm:
            continue
        res = (m,b,c,k, tuple(sorted(basis)), tuple(sorted(translations)))
        results.add(res)
    return results
    
def binexp(x,y):
    h = symbols('h')
    expr = combsimp(binomial(h+4,4)-binomial(h-x+4,4)-binomial(h-y+4,4)+binomial(h-x-y+4,4))
    return expand(expr).evalf()


def satisfy_simple(basis, translations, maxk, c1=None, c2=None, m=None, b=None, c=None):
    if any(e is None for e in [m,b,c]):
        m, _, b, c, _, _, _, _ = run_exps(basis+translations, maxk)

    if c1 is None or c2 is None:
        inters = intersections(basis, translations, maxk)
        x, y = 0, 0
        for i in range(len(inters[1])):
            if len(inters[i][1]) != 0:
                x = i+1
                break
        for i in range(len(inters[2])):
            if len(inters[i][2]) != 0:
                y = i+1
                break
    x = c1 if c1 is not None else x
    y = c2 if c2 is not None else y

    expr = binexp(x,y)
    expected = sympify(f'{m}*h**2 + {b}*h + {c}')
    return simplify(expr - expected) == 0, x, y, expected, expr
    

def filter_dP3_satisfy_simple(basis_depth, maxk, iters):
    res = []
    for m, b, c, k, basis, translations in random_primitive_dPn_exps(3, basis_depth, maxk, iters):
        basis = list(basis)
        translations = list(translations)
        satisfies, x, y, expected, actual = satisfy_simple(basis, translations, maxk, m=m, b=b, c=c)
        if not satisfies:
            polystr = f'exptected: {expected}, for k >= {k}, actual: {actual}'
            res.append((polystr, x, y, basis+translations))
    return res


def all_combinations_binexp(maxx):
    res = []
    for i in range(2, maxx):
        for j in range(i, maxx):
            res.append((binexp(i,j),i,j))
    return res



def magnitue_grows_with_linear_term_dP2():
    for i in range(20):
        A, _ = primitive_triangle_basis(6)
        print(f"\ni = {i}")
        for j in range(8):
            print(single_sumset(A + [(j+3, j+3)]))


"""------------------------WORKPSPACE----------------------------"""

# single_sumset([(0, 2), (1, 0), (1, 1), (9, 8), (11, 11)]) TODO: Why is this not primitive?

# TODO: explain why p(x) = BinExp(c1,c2) where c1/c2 is the index of the first/second self-similarity 
# for any of the sets in filter_dP3_simple_6_15_1000.csv

# TODO: explain why BinExp(3,6) does not work but BinExp(3,7) does for the following set
# print(satisfy_simple([(0,0), (2,1), (1,1)], [(6,3), (1,4)], maxk=15, c1=3, c2=7))
# single_sumset(
#     [(0, 0), (2, 1), (1, 1), (6, 3), (1, 4)],
#     16,
#     basis=[(0, 0), (2, 1), (1, 1)],
#     translations=[(6, 3), (1, 4)],
#     slice=(0,10),
#     plot=True,
#     show_intersections=True,
# )


# TODO: Explain why no expression of the form BinExp(c1,c2) works for the following set
# single_sumset(
#     [(0, 0), (2, 1), (1, 1), (3, 5), (2, 4)],
#     15,
#     basis=[(0, 0), (2, 1), (1, 1)],
#     translations=[(2, 5), (2, 4)],
#     slice=(0,10),
#     plot=True,
#     show_intersections=True,
# )

# write_to_csv(f'random_2d_exps/privimite_{2+n}gons_{maxx}_{iters}.csv', random_primitive_dPn_exps(n,maxx,iters))
# write_to_csv('random_2d_exps/filter_dP3_not_simple_6_15_10.csv', filter_dP3_satisfy_simple(6, 15, 10), legend=('P(h)', 'c_1', 'c_2', 'A'))
# write_to_csv('random_2d_exps/all_comb_binexp_30.csv', all_combinations_binexp(30), legend=('P(x)', 'c1', 'c2'))

# Conj: For every d+3 elements set A, p(x) = BinExp(c1,c2) for some c1,c2 >= 2 where
# BinExp(c1,c2) = choose(h+4,4)-choose(h-c1+4,4)-choose(h-c2+4,4)+choose(h-c1-c2+4,4) 

# Seems like for 1st intersection to be unique we need d+1 simplices for which one of the basis
# points is in the interior of the convex hull. There are only a few other cases if we have d+3
# points: 1) d+1 simplex with all basis points on the boundary, 3) d-simplex with all basis points
# the boundary and a translate in the interior, 4) d-simplex with all basis points on the boundary
# and translate on the boundary. What is the difference between these?

# Q: what happens for d-simplices?

